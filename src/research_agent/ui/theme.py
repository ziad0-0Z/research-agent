"""Gradio theme: maps the design tokens in styles.css onto Gradio's own components (inputs, buttons, tables)."""
from __future__ import annotations

import gradio as gr

INK = "#0E1A30"
SLATE = "#4F5D75"
PAPER = "#F4F6F9"
LINE = "#DDE2EA"
ACCENT = "#2F45D6"
ACCENT_DARK = "#2335B0"


def theme() -> gr.themes.Base:
    font = [gr.themes.GoogleFont("IBM Plex Sans"), "Segoe UI", "system-ui", "sans-serif"]
    mono = [gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "ui-monospace", "monospace"]
    t = gr.themes.Base(primary_hue=gr.themes.colors.indigo, neutral_hue=gr.themes.colors.slate,
                       radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_md,
                       font=font, font_mono=mono)
    return t.set(
        body_background_fill=PAPER,
        body_text_color=INK,
        body_text_color_subdued=SLATE,
        background_fill_primary="#FFFFFF",
        background_fill_secondary=PAPER,
        border_color_primary=LINE,
        block_border_color=LINE,
        block_background_fill="#FFFFFF",
        block_shadow="none",
        block_label_text_color=SLATE,
        block_label_text_weight="500",
        block_title_text_weight="600",
        input_background_fill="#FFFFFF",
        input_border_color=LINE,
        input_border_color_focus=ACCENT,
        input_shadow="none",
        input_shadow_focus=f"0 0 0 3px rgba(47,69,214,.14)",
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_DARK,
        button_primary_text_color="#FFFFFF",
        button_primary_border_color=ACCENT,
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_hover="#EEF1F6",
        button_secondary_border_color=LINE,
        button_secondary_text_color=INK,
        button_cancel_background_fill="#FFFFFF",
        button_cancel_background_fill_hover="#FDECEA",
        button_cancel_text_color="#B42318",
        button_cancel_border_color="#F3C5C0",
        checkbox_background_color_selected=ACCENT,
        checkbox_border_color_selected=ACCENT,
        color_accent=ACCENT,
        color_accent_soft="#EEF0FD",
        table_border_color=LINE,
        table_even_background_fill="#FFFFFF",
        table_odd_background_fill="#FFFFFF",
        table_row_focus="#F1F3FE",
        link_text_color=ACCENT,
        link_text_color_hover=ACCENT_DARK,
    )
