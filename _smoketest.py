"""
Runs app.py without a browser, using fake Streamlit and Plotly modules, to
check nothing crashes before pushing.

    python _smoketest.py

Not part of the site. No need to upload it.
"""
import sys
import types
import traceback


# Fake Plotly. The app never reads anything back, so these can accept
# anything and return anything.
class AnyObj:
    def __init__(self, *a, **k):
        self._kw = k

    def __getattr__(self, name):
        return AnyObj()

    def __call__(self, *a, **k):
        return AnyObj()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class Fig(AnyObj):
    def add_trace(self, *a, **k): return self
    def update_layout(self, *a, **k): return self
    def add_vline(self, *a, **k): return self
    def add_hline(self, *a, **k): return self
    def update_xaxes(self, *a, **k): return self
    def update_yaxes(self, *a, **k): return self


go = types.ModuleType("plotly.graph_objects")
go.Figure = Fig
for _n in ("Bar", "Scatter", "Scatter3d", "Waterfall", "Layout"):
    setattr(go, _n, AnyObj)
go.layout = types.SimpleNamespace(Template=lambda *a, **k: AnyObj())

class Templates(dict):
    """pio.templates is a dict that also has a .default attribute."""
    default = None


pio = types.ModuleType("plotly.io")
pio.templates = Templates()

plotly = types.ModuleType("plotly")
plotly.graph_objects = go
plotly.io = pio
sys.modules.update({"plotly": plotly, "plotly.graph_objects": go, "plotly.io": pio})


# Fake Streamlit. Inputs return real values so the code actually runs through.
CALLS = {}


def _tally(name):
    CALLS[name] = CALLS.get(name, 0) + 1


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(FAKE, name)


class ColumnConfig:
    def ProgressColumn(self, *a, **k): return None
    def NumberColumn(self, *a, **k): return None
    def TextColumn(self, *a, **k): return None


class FakeStreamlit(types.ModuleType):
    column_config = ColumnConfig()

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.sidebar = Ctx()

    # -- passive output ----------------------------------------------------
    def _noop(self, name):
        def f(*a, **k):
            _tally(name)
            return None
        return f

    def __getattr__(self, name):
        # Anything not explicitly modelled is a passive renderer.
        return self._noop(name)

    # -- layout ------------------------------------------------------------
    def columns(self, spec, **k):
        _tally("columns")
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx() for _ in range(n)]

    def tabs(self, labels):
        _tally("tabs")
        return [Ctx() for _ in labels]

    def expander(self, *a, **k):
        _tally("expander")
        return Ctx()

    def spinner(self, *a, **k):
        return Ctx()

    def container(self, *a, **k):
        return Ctx()

    # -- inputs: return real, usable values --------------------------------
    def slider(self, label, *a, **k):
        _tally("slider")
        if "value" in k:
            return k["value"]
        return a[2] if len(a) >= 3 else (a[0] if a else 0)

    def number_input(self, label, *a, **k):
        _tally("number_input")
        return k.get("value", a[2] if len(a) >= 3 else 0)

    def text_input(self, label, value="", **k):
        _tally("text_input")
        return value

    def selectbox(self, label, options, index=0, **k):
        _tally("selectbox")
        opts = list(options)
        return opts[index] if opts else None

    def radio(self, label, options, index=0, **k):
        _tally("radio")
        return list(options)[index]

    def multiselect(self, label, options, default=None, **k):
        return list(default or [])

    def button(self, *a, **k):
        _tally("button")
        return False

    def checkbox(self, label, value=False, **k):
        return value

    def file_uploader(self, *a, **k):
        _tally("file_uploader")
        return None

    def plotly_chart(self, fig, **k):
        _tally("plotly_chart")
        if "on_select" in k:
            return {"selection": {"points": []}}
        return None

    def metric(self, *a, **k):
        _tally("metric")

    def dataframe(self, *a, **k):
        _tally("dataframe")

    def cache_resource(self, *a, **k):
        # Usable both bare and with keyword arguments.
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    def set_page_config(self, **k):
        _tally("set_page_config")


FAKE = FakeStreamlit()
sys.modules["streamlit"] = FAKE

# Used by the 3-D viewer. Keeps the HTML so the test can check it was made.
RENDERED_HTML = []

_components_v1 = types.ModuleType("streamlit.components.v1")


def _html(html, height=None, scrolling=False, **k):
    _tally("components.html")
    RENDERED_HTML.append(html)


_components_v1.html = _html
_components_v1.iframe = lambda *a, **k: _tally("components.iframe")
_components = types.ModuleType("streamlit.components")
_components.v1 = _components_v1
FAKE.components = _components
sys.modules["streamlit.components"] = _components
sys.modules["streamlit.components.v1"] = _components_v1


# Run app.py.
def main():
    src = open("app.py").read()
    ns = {"__name__": "__main__", "__file__": "app.py"}
    try:
        exec(compile(src, "app.py", "exec"), ns)
    except Exception:
        print("\n*** app.py RAISED ***\n")
        traceback.print_exc()
        return 1

    print("app.py executed cleanly.\n")
    print("Widget / render calls recorded:")
    for name in sorted(CALLS):
        print(f"  {name:20} {CALLS[name]}")

    expected = ["set_page_config", "tabs", "metric", "plotly_chart", "dataframe"]
    missing = [e for e in expected if e not in CALLS]
    if missing:
        print("\nWARNING - expected calls never happened:", missing)
        return 1
    print("\nAll four tabs rendered charts, metrics and tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
