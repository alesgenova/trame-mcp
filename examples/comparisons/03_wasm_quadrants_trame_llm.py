"""``03_wasm_quadrants`` rebuilt on trame-llm's MCP utilities — a comparison study.

Run:  python examples/comparisons/03_wasm_quadrants_trame_llm.py --server
Deps: pip install "trame>=3.11" "trame-dataclass>=2.1" vtk trame-vtklocal trame-llm

Connect an agent:
    claude mcp add --transport http wasm-quadrants-llm http://localhost:9000/mcp

The application itself (models, VTK pipelines, reactive UI) is copied verbatim
from ``examples/03_wasm_quadrants.py`` — trame-dataclass is an independent
library, so the only thing that changes here is the MCP layer. What trame-mcp
provided there and what replaces it here:

=====================  =========================================================
trame-mcp version      this port (trame-llm utilities)
=====================  =========================================================
``MCPApp`` mounts       trame-llm has NO endpoint: its FastMCP instance is only
``/mcp`` on the trame   consumed in-process by its chat loop. To drive the app
web server itself       over MCP we must run a SECOND web server (uvicorn) on a
                        second port, reached by digging the FastMCP instance out
                        of the private ``LLMClient._mcps`` list.
4 generic tools         12 hand-written facade tools. Instance addressing,
(``dataclass_*``)       reachability, per-field access policy, serialization
+ ``@tool`` markers     and validation are all re-implemented by hand below
on model methods        (see the "hand-rolled" sections). Every model field is
                        now maintained in three places: the model, the facade
                        serializer/validator, and the tool docstrings.
``dataclass_schemas``   Nothing. The data model is described in prose inside
(machine-generated,     ``instructions`` and tool docstrings; it silently
always current)         drifts when the models change.
``ServerOnly`` fields   A hand-maintained skip-list. Forgetting one field leaks
invisible by policy     a live VTK object into a JSON response (crash).
tool-call lock          None. Safe here only because every facade tool is a
(``serialize_tools``)   single ``async def`` with no awaits inside.
``@prompt`` support     trame-llm's config dict has no "prompts" key; the
                        prompt is registered by reaching into the private
                        FastMCP instance directly.
``--mcp-auth-key`` /    Not wired. FastMCP has its own auth providers, but they
``--mcp-disable`` CLI   are a separate system from the trame app's CLI/session.
deps: none beyond       deps: fastmcp + openai + uvicorn/starlette + ~60
trame itself            transitive packages, even though no LLM key is used.
=====================  =========================================================

Everything below between "BEGIN verbatim" / "END verbatim" markers is identical
to 03_wasm_quadrants.py except that the ``@tool`` / ``@prompt`` decorators are
removed (the methods stay — the UI and the facade both call them).
"""

import contextlib
from typing import Literal

# Required for rendering/interactor factories
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from trame.app import TrameApp
from trame.ui.html import DivLayout
from trame.widgets import client, html, vtklocal
from trame.widgets import dataclass as dc
from trame_dataclass.v2 import (
    ClientOnly,
    ServerOnly,
    StateDataModel,
    Sync,
    get_instance,
    watch,
)
from trame_llm.core import LLMClient
from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkSphereSource,
    vtkSuperquadricSource,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText

OBJECT_TYPES = ("cone", "sphere", "cylinder", "cube", "torus", "text")
MAX_QUADRANTS = 25
ObjectType = Literal["cone", "sphere", "cylinder", "cube", "torus", "text"]
REPRESENTATIONS = ("surface", "wireframe", "points")

# -----------------------------------------------------------------------------
# BEGIN verbatim from 03_wasm_quadrants.py (minus @tool/@prompt decorators)
# -----------------------------------------------------------------------------

# Multiple LocalViews need the async wasm runtime, otherwise only one view's
# event loop runs (trame-vtklocal issue #76). Async requires JSPI
# (WebAssembly.Suspending): available by default on recent Chromium, not on
# Firefox. Detect it client-side and degrade to sync (renders fine, but only
# one quadrant's camera is interactive) instead of crashing.
WASM_CONFIG_EXPR = (
    "(utils.get('WebAssembly') || {}).Suspending "
    "? {rendering: 'webgl', exec: 'async'} "
    ": {rendering: 'webgl', exec: 'sync'}"
)

CELL_STYLE = (
    "border: 1px solid #445; border-radius: 6px; padding: 4px; margin: 0; "
    "flex: 1; display: flex; flex-direction: column; min-height: 0; "
    "font-family: sans-serif; font-size: 14px;"
)
ROW_STYLE = "display: flex; gap: 6px; align-items: center;"

# VTK Set/Get pairs that are internals, not user-facing geometry properties
VTK_PROPERTY_EXCLUDE = {
    "AbortExecute",
    "AbortOutput",
    "Debug",
    "GlobalWarningDisplay",
    "ObjectName",
    "OutputPointsPrecision",
    "Progress",
    "ProgressShiftScale",
    "ProgressText",
    "ReferenceCount",
    "ReleaseDataFlag",
}


def vtk_scalar_properties(obj):
    """Introspect a VTK object for its scalar/vector Set/Get property pairs.

    Returns {name: {"value": ..., "min": ..., "max": ...}} — min/max only for
    properties that advertise bounds (VTK's Get<Name>MinValue convention).
    """
    result = {}
    for attr in dir(obj):
        if not attr.startswith("Set"):
            continue
        name = attr[3:]
        if not name or name in VTK_PROPERTY_EXCLUDE or name.endswith(("MinValue", "MaxValue")):
            continue
        getter = getattr(obj, f"Get{name}", None)
        if getter is None:
            continue
        try:
            value = getter()
        except TypeError:  # getter requires arguments -> not a plain property
            continue
        is_scalar = isinstance(value, (bool, int, float, str))
        is_vector = isinstance(value, tuple) and bool(value) and all(
            isinstance(v, (int, float)) for v in value
        )
        if not (is_scalar or is_vector):
            continue
        entry = {"value": list(value) if is_vector else value}
        for bound, suffix in (("min", "MinValue"), ("max", "MaxValue")):
            bound_getter = getattr(obj, f"Get{name}{suffix}", None)
            if bound_getter is not None:
                with contextlib.suppress(TypeError):
                    entry[bound] = bound_getter()
        result[name] = entry
    return result


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


class SceneObject(StateDataModel):
    """Base class for anything that can live in a Scene."""

    type = ClientOnly(str, "")  # read-only over MCP by field mode
    color = Sync(list[float], lambda: [0.9, 0.9, 0.9])
    position = Sync(list[float], lambda: [0.0, 0.0, 0.0])
    scale = Sync(list[float], lambda: [1.0, 1.0, 1.0])
    rotation = Sync(list[float], lambda: [0.0, 0.0, 0.0])  # degrees around x/y/z
    opacity = Sync(float, 1.0)  # 0 (transparent) - 1 (opaque)
    representation = Sync(str, "surface")  # surface | wireframe | points
    visible = Sync(bool, True)

    source = ServerOnly(object)
    actor = ServerOnly(object)
    scene = ServerOnly(object)  # backref, not a dataclass field -> no MCP cycle

    SOURCE_CLASS = None  # set by concrete subclasses

    @classmethod
    def generate_gui(cls, trame_server=None) -> str:
        with html.Div(
            trame_server=trame_server,
            v_if="self_available && self._id",
            style=ROW_STYLE,
        ) as root:
            cls._gui_row(root)
        return root.html

    @classmethod
    def _gui_row(cls, _root):
        html.Span("{{ self.type }} #{{ self._id }}", style="min-width: 5.5em;")
        html.Span("visible")
        html.Input(type="checkbox", v_model="self.visible")
        html.Button("remove", click=(cls.ui_remove, "[self._id]"))

    @classmethod
    def ui_remove(cls, object_id):
        obj = get_instance(object_id)
        if obj is not None and obj.scene is not None:
            obj.scene.remove_object(object_id)

    def vtk_source_properties(self) -> dict:
        """List every scalar/vector property of this object's VTK source."""
        if self.source is None:
            msg = "Object has no VTK source yet"
            raise RuntimeError(msg)
        return vtk_scalar_properties(self.source)

    def set_vtk_source_property(self, name: str, value) -> str:
        """Set a VTK source property by name (see vtk_source_properties)."""
        properties = vtk_scalar_properties(self.source)
        if name not in properties:
            msg = (
                f"Unknown VTK property {name!r} on {type(self.source).__name__}. "
                f"Available: {sorted(properties)}"
            )
            raise ValueError(msg)
        setter = getattr(self.source, f"Set{name}")
        if isinstance(value, list):
            setter(*value)
        else:
            setter(value)
        self.scene.request_render()
        return f"{type(self.source).__name__}.{name} = {value}"

    def build(self, scene):
        self.scene = scene
        self.source = self.SOURCE_CLASS()
        self._setup_source()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self.source.GetOutputPort())
        self.actor = vtkActor()
        self.actor.SetMapper(mapper)
        self._apply_resolution()
        self._apply_appearance()
        return self.actor

    def _setup_source(self):
        """Subclass hook to configure the freshly created source."""

    def _apply_resolution(self):
        """No resolution on the base class (see ResolutionObject)."""

    def _apply_appearance(self):
        prop = self.actor.GetProperty()
        prop.SetColor(*self.color[:3])
        prop.SetOpacity(max(0.0, min(float(self.opacity), 1.0)))
        representation = str(self.representation).lower()
        if representation == "wireframe":
            prop.SetRepresentationToWireframe()
        elif representation == "points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(4)
        else:
            prop.SetRepresentationToSurface()
        self.actor.SetPosition(*self.position[:3])
        self.actor.SetScale(*self.scale[:3])
        self.actor.SetOrientation(*self.rotation[:3])
        self.actor.SetVisibility(bool(self.visible))

    @watch(
        "color", "position", "scale", "rotation", "opacity", "representation",
        "visible", sync=True,
    )
    def _on_appearance(self, *_):
        if self.actor is None:
            return
        self._apply_appearance()
        self.scene.request_render()


class ResolutionObject(SceneObject):
    """Scene object with a tessellation resolution (cone, cylinder, sphere)."""

    resolution = Sync(int, 16)

    @classmethod
    def _gui_row(cls, _root):
        html.Span("{{ self.type }} #{{ self._id }}", style="min-width: 5.5em;")
        html.Span("visible")
        html.Input(type="checkbox", v_model="self.visible")
        html.Span("resolution")
        html.Input(
            type="range", min=3, max=64, v_model_number="self.resolution",
            style="flex: 1;",
        )
        html.Button("remove", click=(cls.ui_remove, "[self._id]"))

    def _apply_resolution(self):
        self.source.SetResolution(max(3, min(int(self.resolution), 128)))

    @watch("resolution", sync=True)
    def _on_resolution(self, _resolution):
        if self.source is None:
            return
        self._apply_resolution()
        self.scene.request_render()


class ConeObject(ResolutionObject):
    type = ClientOnly(str, "cone")
    SOURCE_CLASS = vtkConeSource


class CylinderObject(ResolutionObject):
    type = ClientOnly(str, "cylinder")
    SOURCE_CLASS = vtkCylinderSource


class SphereObject(ResolutionObject):
    type = ClientOnly(str, "sphere")
    SOURCE_CLASS = vtkSphereSource

    def _apply_resolution(self):
        resolution = max(3, min(int(self.resolution), 128))
        self.source.SetThetaResolution(resolution)
        self.source.SetPhiResolution(resolution)


class CubeObject(SceneObject):
    type = ClientOnly(str, "cube")
    SOURCE_CLASS = vtkCubeSource


class TorusObject(ResolutionObject):
    type = ClientOnly(str, "torus")
    thickness = Sync(float, 0.3)  # ring thickness, 0-1
    SOURCE_CLASS = vtkSuperquadricSource

    def _setup_source(self):
        self.source.ToroidalOn()
        self.source.SetThickness(max(0.01, min(float(self.thickness), 1.0)))

    def _apply_resolution(self):
        resolution = max(3, min(int(self.resolution), 128))
        self.source.SetThetaResolution(resolution)
        self.source.SetPhiResolution(resolution)

    @watch("thickness", sync=True)
    def _on_thickness(self, thickness):
        if self.source is None:
            return
        self.source.SetThickness(max(0.01, min(float(thickness), 1.0)))
        self.scene.request_render()


class TextObject(SceneObject):
    type = ClientOnly(str, "text")
    text = Sync(str, "hello")
    SOURCE_CLASS = vtkVectorText

    def _setup_source(self):
        self.source.SetText(self.text or " ")

    @classmethod
    def _gui_row(cls, _root):
        html.Span("{{ self.type }} #{{ self._id }}", style="min-width: 5.5em;")
        html.Span("visible")
        html.Input(type="checkbox", v_model="self.visible")
        html.Input(v_model="self.text", style="flex: 1;")
        html.Button("remove", click=(cls.ui_remove, "[self._id]"))

    @watch("text", sync=True)
    def _on_text(self, text):
        if self.source is None:
            return
        self.source.SetText(str(text) or " ")
        self.scene.request_render()


TYPE_TO_CLASS = {
    "cone": ConeObject,
    "sphere": SphereObject,
    "cylinder": CylinderObject,
    "cube": CubeObject,
    "torus": TorusObject,
    "text": TextObject,
}


class Scene(StateDataModel):
    name = Sync(str, "Scene")
    background = Sync(list[float], lambda: [0.12, 0.12, 0.18])
    objects = Sync(list[SceneObject], list, has_dataclass=True)
    overlay_open = ClientOnly(bool, True)  # per-browser UI state, never read back

    renderer = ServerOnly(object)
    render_window = ServerOnly(object)
    interactor = ServerOnly(object)
    view = ServerOnly(object)

    @classmethod
    def generate_gui(cls, trame_server=None) -> str:
        with html.Div(
            trame_server=trame_server,
            v_if="self_available && self._id",
            style=CELL_STYLE,
        ) as root:
            # header: editable name + add-object buttons + camera reset
            with html.Div(style=ROW_STYLE):
                html.Input(v_model="self.name", style="width: 6em;")
                for object_type in OBJECT_TYPES:
                    html.Button(
                        f"+{object_type}",
                        click=(cls.ui_add_object, f"[self._id, '{object_type}']"),
                    )
                html.Button("reset view", click=(cls.ui_reset_camera, "[self._id]"))

            # the WASM view lives in a per-scene server template, created once
            # in build_pipeline() and referenced reactively by name here
            with html.Div(style="flex: 1; min-height: 100px; position: relative;"):
                client.ServerTemplate(name=("`scene_view_${self._id}`",))

                # object rows float over the top of the 3D view; minimizable,
                # capped to the quadrant height, scrollable when crowded
                with html.Div(
                    style=(
                        "position: absolute; top: 4px; left: 4px; z-index: 1; "
                        "max-height: calc(100% - 8px); "
                        "max-width: calc(100% - 8px); "
                        "display: flex; flex-direction: column; "
                        "background: rgba(255, 255, 255, 0.65); "
                        "backdrop-filter: blur(2px); "
                        "border-radius: 4px; padding: 2px 6px;"
                    ),
                    v_if="self.objects.length",
                ):
                    with html.Div(style=ROW_STYLE):
                        html.Span(
                            "objects ({{ self.objects.length }})",
                            style="font-weight: bold;",
                        )
                        html.Button(
                            "{{ self.overlay_open ? '-' : '+' }}",
                            click="self.overlay_open = !self.overlay_open",
                            style="margin-left: auto;",
                        )
                    with html.Div(
                        v_show="self.overlay_open",
                        style=(
                            "overflow-y: auto; min-height: 0; "
                            "display: flex; flex-direction: column; gap: 2px;"
                        ),
                    ):
                        with html.Div(v_for="obj in self.objects", key=("obj._id",)):
                            dc.Gui(instance=("obj._id",))
        return root.html

    @classmethod
    def ui_add_object(cls, scene_id, object_type):
        scene = get_instance(scene_id)
        if scene is not None:
            scene.add_object(object_type)

    @classmethod
    def ui_reset_camera(cls, scene_id):
        scene = get_instance(scene_id)
        if scene is not None:
            scene.reset_camera()

    def _next_free_position(self):
        # First default slot along X not occupied by an existing object, so a
        # new object never overlaps survivors of earlier removals
        taken = {round(o.position[0] / 1.2) for o in self.objects if o is not None}
        slot = 0
        while slot in taken:
            slot += 1
        return [slot * 1.2, 0.0, 0.0]

    def build_pipeline(self):
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(*self.background[:3])
        self.render_window = vtkRenderWindow()
        self.render_window.OffScreenRenderingOn()
        self.render_window.AddRenderer(self.renderer)
        # The WASM remote session requires an interactor on the mirrored window
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # One LocalView per scene, in its own named template, created exactly
        # once - object add/remove never remounts it
        with DivLayout(
            self.server,
            template_name=f"scene_view_{self._id}",
            style="position: absolute; inset: 0;",
        ):
            self.view = vtklocal.LocalView(
                self.render_window,
                throttle_rate=20,
                config=(WASM_CONFIG_EXPR,),
            )

    def request_render(self, push_camera=False):
        if self.render_window is None:
            return
        self.render_window.Render()
        if self.view is not None:
            if push_camera:
                self.view.update(push_camera=True)
            else:
                self.view.update_throttle()

    @watch("background", sync=True)
    def _on_background(self, background):
        if self.renderer is None:
            return
        self.renderer.SetBackground(*background[:3])
        self.request_render()

    def add_object(self, type: str) -> dict:
        """Add a new object to this scene; returns {"id", "type"}."""
        if type not in TYPE_TO_CLASS:
            msg = f"Unknown type {type!r}. Available types: {list(TYPE_TO_CLASS)}"
            raise ValueError(msg)

        obj = TYPE_TO_CLASS[type](self.server)
        obj.position = self._next_free_position()
        self.renderer.AddActor(obj.build(self))
        self.objects = [*self.objects, obj]

        self.renderer.ResetCamera()
        self.request_render(push_camera=True)
        return {"id": obj._id, "type": type}

    def remove_object(self, object_id: str) -> str:
        """Remove an object from this scene by its id."""
        live = [o for o in self.objects if o is not None]
        match = [o for o in live if o._id == object_id]
        if not match:
            msg = (
                f"No object {object_id!r} in scene {self.name!r}. "
                f"Object ids: {[o._id for o in live]}"
            )
            raise ValueError(msg)
        self.renderer.RemoveActor(match[0].actor)
        self.objects = [o for o in live if o._id != object_id]
        self.request_render()
        return f"Removed {object_id}"

    def clear_objects(self) -> str:
        """Remove every object from this scene."""
        self.renderer.RemoveAllViewProps()
        self.objects = []
        self.request_render()
        return "Scene cleared"

    def reset_camera(self) -> str:
        """Reset this scene's camera to fit all visible objects."""
        self.renderer.ResetCamera()
        self.request_render(push_camera=True)
        return "Camera reset"


class Workspace(StateDataModel):
    rows = Sync(int, 1)
    cols = Sync(int, 1)
    scenes = Sync(list[Scene], list, has_dataclass=True)

    def set_grid(self, rows: int, cols: int) -> str:
        """Set the quadrant grid to rows x cols (max 25 quadrants total)."""
        if rows < 1 or cols < 1 or rows * cols > MAX_QUADRANTS:
            msg = f"Grid must be at least 1x1 and at most {MAX_QUADRANTS} quadrants (got {rows}x{cols})"
            raise ValueError(msg)
        self.rows = rows
        self.cols = cols
        return f"Grid set to {rows}x{cols} ({len(self.scenes)} scenes)"

    @watch("rows", "cols", sync=True)
    def _on_grid(self, rows, cols):
        # Clamp direct field writes (UI / facade)
        rows, cols = max(1, int(rows)), max(1, int(cols))
        while rows * cols > MAX_QUADRANTS:
            if cols > rows:
                cols -= 1
            else:
                rows -= 1
        self.rows, self.cols = rows, cols  # no-op when already equal
        self.reconcile()

    def reconcile(self):
        desired = self.rows * self.cols
        scenes = list(self.scenes)

        while len(scenes) > desired:
            scenes.pop()
        while len(scenes) < desired:
            scene = Scene(self.server, name=f"Scene {len(scenes) + 1}")
            scene.build_pipeline()
            scenes.append(scene)

        if [s._id for s in scenes] != [s._id for s in self.scenes]:
            self.scenes = scenes


# -----------------------------------------------------------------------------
# END verbatim from 03_wasm_quadrants.py
# -----------------------------------------------------------------------------
#
# Everything below is the code that trame-mcp made unnecessary.
#
# -----------------------------------------------------------------------------
# Hand-rolled field policy (trame-mcp: derived from Sync/ClientOnly/ServerOnly)
# -----------------------------------------------------------------------------

# Which fields exist, per type, and which are writable. This table shadows the
# model declarations above and must be kept in sync with them BY HAND. With
# trame-mcp the same information was derived from the descriptors themselves.
OBJECT_READABLE = (
    "type", "color", "position", "scale", "rotation", "opacity",
    "representation", "visible",
)
OBJECT_WRITABLE = (
    "color", "position", "scale", "rotation", "opacity",
    "representation", "visible",
)  # `type` is ClientOnly -> read-only; source/actor/scene are ServerOnly ->
#    they must NEVER appear in a response (live VTK objects, not JSON-safe)
TYPE_EXTRA_FIELDS = {
    "cone": ("resolution",),
    "sphere": ("resolution",),
    "cylinder": ("resolution",),
    "cube": (),
    "torus": ("resolution", "thickness"),
    "text": ("text",),
}
SCENE_WRITABLE = ("name", "background")  # overlay_open is ClientOnly


class QuadrantsMCPController:
    """The static MCP facade, in trame-llm's ``controller.mcp`` idiom.

    trame-llm/FastMCP tools are plain functions registered once, up front.
    There is no instance addressing, so every tool below takes explicit
    ``scene_id`` / ``object_id`` parameters and re-implements resolution,
    reachability, serialization and validation that the ``dataclass_*``
    generic tools provided for free.

    All tools are ``async def`` on purpose: FastMCP runs *sync* tools in a
    worker thread, which is unsafe for trame state and VTK. Async tools run
    inline on the serving event loop, which (because we start uvicorn on
    trame's own loop, below) is the trame loop. None of this is enforced
    anywhere — it is a convention this file must uphold by hand.
    """

    def __init__(self, app):
        self.app = app

    @property
    def workspace(self):
        return self.app.workspace

    # -- the trame-llm registration contract ---------------------------------

    @property
    def mcp(self):
        return {
            "name": "wasm-quadrants-llm",
            "instructions": (
                "This app shows a grid of 3D scenes rendered in the browser. "
                "Data model (prose only - there is no schema tool): a workspace "
                "has rows x cols scenes; each scene has a name, a background "
                "color [r,g,b in 0-1] and a list of objects. Objects have "
                "color [r,g,b in 0-1], position [x,y,z], scale, rotation "
                "[degrees around x,y,z], opacity [0-1], representation "
                "(surface | wireframe | points) and visible. Cones, spheres, "
                "cylinders and toruses also have an integer resolution; "
                "toruses have thickness [0-1]; text objects have a text "
                "string. Start with describe_workspace, then use set_grid, "
                "add_object and update_object."
            ),
            "tools": [
                self.describe_workspace,
                self.get_scene,
                self.get_object,
                self.set_grid,
                self.update_scene,
                self.add_object,
                self.remove_object,
                self.clear_objects,
                self.reset_camera,
                self.update_object,
                self.vtk_source_properties,
                self.set_vtk_source_property,
            ],
        }

    # -- hand-rolled: reachability + resolution (trame-mcp: DataclassLayer) --

    def _live_scenes(self):
        return {s._id: s for s in self.workspace.scenes if s is not None}

    def _live_objects(self):
        return {
            o._id: o
            for s in self.workspace.scenes if s is not None
            for o in s.objects if o is not None
        }

    def _resolve_scene(self, scene_id) -> Scene:
        # get_instance(scene_id) alone would resolve ANY dataclass instance in
        # the process, including removed ones - restrict to the workspace by
        # hand.
        scenes = self._live_scenes()
        if scene_id not in scenes:
            msg = (
                f"Unknown scene {scene_id!r}. "
                f"Scene ids: {sorted(scenes)}. Use describe_workspace."
            )
            raise ValueError(msg)
        return scenes[scene_id]

    def _resolve_object(self, object_id) -> SceneObject:
        objects = self._live_objects()
        if object_id not in objects:
            msg = (
                f"Unknown object {object_id!r}. "
                f"Object ids: {sorted(objects)}. Use describe_workspace."
            )
            raise ValueError(msg)
        return objects[object_id]

    # -- hand-rolled: serialization (trame-mcp: dataclass_get) ---------------

    def _serialize_object(self, obj):
        fields = OBJECT_READABLE + TYPE_EXTRA_FIELDS[obj.type]
        return {"id": obj._id, **{name: getattr(obj, name) for name in fields}}

    def _serialize_scene(self, scene, with_objects=True):
        data = {
            "id": scene._id,
            "name": scene.name,
            "background": scene.background,
        }
        if with_objects:
            data["objects"] = [
                self._serialize_object(o) for o in scene.objects if o is not None
            ]
        else:
            data["objects"] = [
                {"id": o._id, "type": o.type}
                for o in scene.objects if o is not None
            ]
        return data

    # -- tools ----------------------------------------------------------------

    async def describe_workspace(self) -> dict:
        """Describe the whole workspace: grid size, scenes and their objects.

        Returns {rows, cols, scenes: [{id, name, background, objects:
        [{id, type}]}]}. Call this first to discover scene and object ids.
        """
        return {
            "rows": self.workspace.rows,
            "cols": self.workspace.cols,
            "scenes": [
                self._serialize_scene(s, with_objects=False)
                for s in self.workspace.scenes if s is not None
            ],
        }

    async def get_scene(self, scene_id: str) -> dict:
        """Read one scene in full: name, background and every object with all
        of its fields. `scene_id` comes from describe_workspace."""
        return self._serialize_scene(self._resolve_scene(scene_id))

    async def get_object(self, object_id: str) -> dict:
        """Read one object with all of its fields. `object_id` comes from
        describe_workspace or get_scene."""
        return self._serialize_object(self._resolve_object(object_id))

    async def set_grid(self, rows: int, cols: int) -> str:
        """Set the quadrant grid to rows x cols (max 25 quadrants total).

        Each quadrant holds one scene; shrinking the grid destroys trailing
        scenes and their objects.
        """
        return self.workspace.set_grid(rows, cols)

    async def update_scene(
        self,
        scene_id: str,
        name: str | None = None,
        background: list[float] | None = None,
    ) -> dict:
        """Update a scene's name and/or background color [r,g,b in 0-1].

        (Other scene attributes are not writable.)
        """
        scene = self._resolve_scene(scene_id)
        if name is not None:
            scene.name = name
        if background is not None:
            scene.background = background
        return self._serialize_scene(scene, with_objects=False)

    async def add_object(self, scene_id: str, type: ObjectType) -> dict:
        """Add a new object (cone, sphere, cylinder, cube, torus or text) to a
        scene. Returns {"id", "type"}; use update_object on the id to change
        its appearance."""
        return self._resolve_scene(scene_id).add_object(type)

    async def remove_object(self, object_id: str) -> str:
        """Remove an object (by id) from the scene that owns it."""
        obj = self._resolve_object(object_id)
        return obj.scene.remove_object(object_id)

    async def clear_objects(self, scene_id: str) -> str:
        """Remove every object from a scene."""
        return self._resolve_scene(scene_id).clear_objects()

    async def reset_camera(self, scene_id: str) -> str:
        """Reset a scene's camera to fit all visible objects."""
        return self._resolve_scene(scene_id).reset_camera()

    async def update_object(
        self,
        object_id: str,
        color: list[float] | None = None,
        position: list[float] | None = None,
        scale: list[float] | None = None,
        rotation: list[float] | None = None,
        opacity: float | None = None,
        representation: str | None = None,
        visible: bool | None = None,
        resolution: int | None = None,
        thickness: float | None = None,
        text: str | None = None,
    ) -> dict:
        """Update fields of one object; only pass the fields to change.

        color [r,g,b in 0-1], position [x,y,z], scale [sx,sy,sz], rotation
        [degrees around x,y,z], opacity [0-1], representation ("surface",
        "wireframe" or "points"), visible. Type-specific: resolution (cone,
        sphere, cylinder, torus), thickness [0-1] (torus), text (text).
        Passing a field the object's type does not have is an error and
        nothing is applied.
        """
        # The signature above is the union of every field of every object
        # type: it must grow (here, in the docstring, and in the validation
        # below) every time a model gains a field. trame-mcp derived all of
        # this from the model classes.
        obj = self._resolve_object(object_id)
        requested = {
            name: value
            for name, value in {
                "color": color,
                "position": position,
                "scale": scale,
                "rotation": rotation,
                "opacity": opacity,
                "representation": representation,
                "visible": visible,
                "resolution": resolution,
                "thickness": thickness,
                "text": text,
            }.items()
            if value is not None
        }
        if not requested:
            msg = "No fields to update were provided."
            raise ValueError(msg)

        allowed = set(OBJECT_WRITABLE) | set(TYPE_EXTRA_FIELDS[obj.type])
        rejected = sorted(set(requested) - allowed)
        if rejected:
            msg = (
                f"Nothing was applied. Fields {rejected} do not exist on a "
                f"{obj.type!r} object. Writable fields: {sorted(allowed)}"
            )
            raise ValueError(msg)

        for name, value in requested.items():
            setattr(obj, name, value)
        return self._serialize_object(obj)

    async def vtk_source_properties(self, object_id: str) -> dict:
        """Advanced: list every scalar/vector property of an object's
        underlying VTK source, beyond the declared fields.

        Returns {name: {value, min?, max?}}; set them with
        set_vtk_source_property. Prefer update_object when a declared field
        covers the same thing (e.g. resolution).
        """
        return self._resolve_object(object_id).vtk_source_properties()

    async def set_vtk_source_property(
        self, object_id: str, name: str, value: float | int | bool | str | list[float]
    ) -> str:
        """Advanced: set a VTK source property by name, as listed by
        vtk_source_properties (e.g. Height or Radius on a cone).

        Vector-valued properties take a list. Changes render immediately but
        are not tracked as model fields.
        """
        return self._resolve_object(object_id).set_vtk_source_property(name, value)


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------


class WasmQuadrantsApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        self.server.cli.add_argument(
            "--mcp-port",
            type=int,
            default=9000,
            help="Port for the bolt-on FastMCP HTTP server (trame-llm has no endpoint)",
        )
        args, _ = self.server.cli.parse_known_args()
        self.mcp_port = args.mcp_port

        self.workspace = Workspace(self.server)

        # trame-llm's MCP registration: an LLMClient is required even though
        # no LLM is involved here - it is the only public owner of the FastMCP.
        self.llm_client = LLMClient()
        self.llm_client.register_mcp(QuadrantsMCPController(self))

        # trame-llm keeps the FastMCP instance private and never serves it;
        # reach into _mcps to expose it over HTTP ourselves.
        self._fastmcp = self.llm_client._mcps[0]

        # The controller.mcp dict has no "prompts" key (unsupported by
        # register_mcp), so the prompt is registered behind trame-llm's back.
        self._fastmcp.prompt(self.demo_script)

        self.workspace.reconcile()  # initial 1x1 grid
        self._build_ui()  # built once - everything after this is reactive

        self.ctrl.on_server_ready.add_task(self._serve_mcp)

    def demo_script(self) -> str:
        """Suggested walkthrough of the quadrants app."""
        return (
            "Create a 2x2 grid of scenes. Put a red cone in the first scene, "
            "a blue sphere in the second, a green cylinder in the third and a "
            "yellow cube in the last one, then describe the workspace."
        )

    async def _serve_mcp(self, **_):
        # Second web server (uvicorn/starlette) next to trame's aiohttp one.
        # Started on trame's own event loop so that the async facade tools run
        # on the trame loop and can safely touch state and VTK.
        # Caveat: uvicorn installs its own SIGINT handler on the main thread,
        # so the first Ctrl+C stops the MCP server and the second stops trame.
        url = f"http://localhost:{self.mcp_port}/mcp"
        print(f"\nMCP server available at {url}")
        print(f"  claude mcp add --transport http wasm-quadrants-llm {url}\n", flush=True)
        await self._fastmcp.run_http_async(
            show_banner=False,
            transport="http",
            host="localhost",
            port=self.mcp_port,
            path="/mcp",
            stateless_http=True,
            json_response=True,
            log_level="warning",
        )

    def _ui_adjust_grid(self, d_rows, d_cols):
        ws = self.workspace
        rows = max(1, ws.rows + d_rows)
        cols = max(1, ws.cols + d_cols)
        if rows * cols <= MAX_QUADRANTS:
            ws.rows, ws.cols = rows, cols  # sync watcher reconciles

    def _build_ui(self):
        with DivLayout(
            self.server,
            style="height: 100vh; display: flex; flex-direction: column; font-family: sans-serif;",
        ):
            with self.workspace.provide_as("w"):
                with html.Div(style=ROW_STYLE + "padding: 6px;"):
                    html.B("Grid {{ w.rows }} x {{ w.cols }}")
                    html.Span("rows:")
                    html.Button("-", click=(self._ui_adjust_grid, "[-1, 0]"))
                    html.Button("+", click=(self._ui_adjust_grid, "[1, 0]"))
                    html.Span("cols:")
                    html.Button("-", click=(self._ui_adjust_grid, "[0, -1]"))
                    html.Button("+", click=(self._ui_adjust_grid, "[0, 1]"))
                    html.Span(
                        "({{ w.scenes.length }} scenes, max " + str(MAX_QUADRANTS) + ")"
                    )

                grid_style = (
                    "`flex: 1; display: grid; gap: 6px; padding: 6px; min-height: 0; "
                    "grid-template-columns: repeat(${w.cols}, 1fr); "
                    "grid-template-rows: repeat(${w.rows}, 1fr);`"
                )
                with html.Div(style=(grid_style,)):
                    with html.Div(
                        v_for="scene in w.scenes",
                        key=("scene._id",),
                        style="min-height: 0; display: flex;",
                    ):
                        dc.Gui(instance=("scene._id",), style="flex: 1; display: flex;")


def main():
    app = WasmQuadrantsApp()
    app.server.start()


if __name__ == "__main__":
    main()
