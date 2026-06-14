"""Font utilities for plot styling."""

import importlib.resources as pkg_resources

import matplotlib.font_manager as fm

ASSETS_PATH = pkg_resources.files("openretailscience") / "assets"
FONTS_PATH = ASSETS_PATH / "fonts"

# Built-in font registry - maps names to bundled font files
_BUILTIN_FONTS = {
    "poppins_bold": FONTS_PATH / "Poppins-Bold.ttf",
    "poppins_semi_bold": FONTS_PATH / "Poppins-SemiBold.ttf",
    "poppins_regular": FONTS_PATH / "Poppins-Regular.ttf",
    "poppins_medium": FONTS_PATH / "Poppins-Medium.ttf",
    "poppins_light_italic": FONTS_PATH / "Poppins-LightItalic.ttf",
}

# Font cache for performance
_font_cache: dict[str, fm.FontProperties] = {}


def get_font_properties(font_name: str) -> fm.FontProperties:
    """Get matplotlib FontProperties with flexible font resolution.

    This function resolves fonts in the following priority order:
    1. Check cache for previously loaded font
    2. Try to load from bundled fonts if font_name matches a built-in font
    3. Try to load as a system font family name
    4. Raise ValueError if font cannot be found (strict validation)

    Args:
        font_name (str): The name of the font to load. Can be:
            - A built-in font name (e.g., 'poppins_regular')
            - A system font family name (e.g., 'Arial', 'Times New Roman')

    Returns:
        FontProperties: A matplotlib FontProperties object that can be used
            for text rendering.

    Raises:
        ValueError: If the font cannot be found in bundled fonts or system fonts.
    """
    if font_name in _font_cache:
        return _font_cache[font_name]

    # Try to load built-in bundled fonts first
    if font_name in _BUILTIN_FONTS:
        font_props = fm.FontProperties(fname=str(_BUILTIN_FONTS[font_name]))
        _font_cache[font_name] = font_props
        return font_props

    # Try to load as system font family - validate it exists
    font_props = fm.FontProperties(family=font_name)
    try:
        fm.fontManager.findfont(font_props, fallback_to_default=False)
    except ValueError as e:
        error_msg = f"Font '{font_name}' not found. Available bundled fonts: {list(_BUILTIN_FONTS.keys())}"
        raise ValueError(error_msg) from e

    _font_cache[font_name] = font_props
    return font_props
