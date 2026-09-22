"""
Auto-generates a Settings-tab form from a detector/classifier Params model.

This is what lets the Settings tab show one fully-featured sub-tab per
registered detector/classifier (see AbstractDetector.all() /
AbstractClassifier.all() in squeak_peek.plugins) without any hand-written
per-field code: a Params pydantic model describes its own form entirely
through its field types and metadata.

Field metadata conventions (all optional, set via ``Field(..., json_schema_extra={...})``):

    unit          suffix shown on a spin box, e.g. "Hz", "s", "frames"
    caption       muted helper line shown under the row
    decimals      float display precision (default 4)
    step          spin box single-step increment
    group         two fields sharing the same group value (e.g. a
                  min/max frequency pair) get ONE combined range
                  indicator instead of two separate ones
    widget        "file" renders a QLineEdit + Browse... button
    file_filter   the Qt file-dialog filter string, used with widget="file"
    choices       a list of allowed values -> renders as a QComboBox
    label         override the row's label text (default: the field name)

Numeric bounds come from the field's own ``ge``/``le`` (or ``gt``/``lt``)
constraints, which pydantic also uses for validation -- so the form's
range and the model's validation can never drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

FieldRowFn = Callable[..., QWidget]
RangeIndicatorFn = Callable[[QFormLayout, QWidget, QWidget], None]


def _extra(info: FieldInfo) -> dict[str, Any]:
    extra = info.json_schema_extra
    return extra if isinstance(extra, dict) else {}


def _bounds(info: FieldInfo) -> tuple[float | None, float | None]:
    lo = hi = None
    for constraint in info.metadata:
        if hasattr(constraint, "ge") and constraint.ge is not None:
            lo = constraint.ge
        if hasattr(constraint, "gt") and constraint.gt is not None:
            lo = constraint.gt
        if hasattr(constraint, "le") and constraint.le is not None:
            hi = constraint.le
        if hasattr(constraint, "lt") and constraint.lt is not None:
            hi = constraint.lt
    return lo, hi


def build_params_form(
    form: QFormLayout,
    params: BaseModel,
    field_row: FieldRowFn,
    range_indicator: RangeIndicatorFn,
) -> dict[str, QWidget]:
    """Build one form row per field of ``params``, returning
    ``{field_name: widget}`` for read_params_form()/reload_params_form().

    ``field_row(form, label_text, widget, tooltip, caption=None, indicator=True)``
    and ``range_indicator(form, lo_widget, hi_widget)`` are the same
    row-building helpers SettingsTab already uses for its hand-written
    sections (see _tab_settings.py's _field()/_range_indicator()).
    """
    widgets: dict[str, QWidget] = {}
    model_fields = type(params).model_fields
    group_first: dict[str, str] = {}

    for name, info in model_fields.items():
        extra = _extra(info)
        value = getattr(params, name)
        label_text = extra.get("label", name)
        tooltip = info.description or name
        caption = extra.get("caption")
        unit = extra.get("unit", "")
        group = extra.get("group")

        if extra.get("widget") == "file":
            widgets[name] = _build_file_row(form, label_text, tooltip, str(value), extra)
            continue

        if extra.get("choices") is not None:
            widgets[name] = _build_choice_row(form, field_row, label_text, tooltip, caption, value, extra["choices"])
            continue

        if info.annotation is bool:
            widgets[name] = _build_bool_row(form, field_row, label_text, tooltip, caption, value)
            continue

        if info.annotation is str:
            edit = QLineEdit(str(value))
            field_row(form, f"{label_text}:", edit, tooltip, caption, indicator=False)
            widgets[name] = edit
            continue

        # Numeric (int/float)
        lo, hi = _bounds(info)
        is_float = info.annotation is float
        spin: QSpinBox | QDoubleSpinBox = QDoubleSpinBox() if is_float else QSpinBox()
        spin.setRange(
            lo if lo is not None else (-1e12 if is_float else -2_147_483_648),
            hi if hi is not None else (1e12 if is_float else 2_147_483_647),
        )
        if is_float:
            decimals = extra.get("decimals", 4)
            spin.setDecimals(decimals)
            spin.setSingleStep(extra.get("step", 10 ** -decimals))
        if unit:
            spin.setSuffix(f" {unit}")
        spin.setValue(value)

        show_indicator = True
        if group is not None:
            if group not in group_first:
                group_first[group] = name
                show_indicator = False
            else:
                show_indicator = False  # combined indicator added below instead

        field_row(form, f"{label_text}:", spin, tooltip, caption, indicator=show_indicator)
        widgets[name] = spin

        if group is not None and group_first[group] != name:
            range_indicator(form, widgets[group_first[group]], spin)

    return widgets


def _build_file_row(
    form: QFormLayout, label_text: str, tooltip: str, value: str, extra: dict[str, Any],
) -> QLineEdit:
    row = QHBoxLayout()
    edit = QLineEdit(value)
    edit.setToolTip(tooltip)
    browse_btn = QPushButton("Browse…")
    file_filter = extra.get("file_filter", "All files (*)")
    browse_btn.setToolTip(f"Choose a file ({file_filter}).")

    def _browse() -> None:
        path, _ = QFileDialog.getOpenFileName(browse_btn, f"Select {label_text}", "", file_filter)
        if path:
            edit.setText(path)

    browse_btn.clicked.connect(_browse)
    row.addWidget(edit)
    row.addWidget(browse_btn)

    label = QLabel(f"{label_text}:")
    label.setToolTip(tooltip)
    form.addRow(label, row)
    return edit


def _build_choice_row(
    form: QFormLayout, field_row: FieldRowFn, label_text: str, tooltip: str,
    caption: str | None, value: Any, choices: list[Any],
) -> QComboBox:
    combo = QComboBox()
    for choice in choices:
        combo.addItem(str(choice), choice)
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    field_row(form, f"{label_text}:", combo, tooltip, caption, indicator=False)
    return combo


def _build_bool_row(
    form: QFormLayout, field_row: FieldRowFn, label_text: str, tooltip: str,
    caption: str | None, value: bool,
) -> QComboBox:
    combo = QComboBox()
    combo.addItem("Yes", True)
    combo.addItem("No", False)
    combo.setCurrentIndex(0 if value else 1)
    field_row(form, f"{label_text}:", combo, tooltip, caption, indicator=False)
    return combo


def read_params_form(params_cls: type[BaseModel], widgets: dict[str, QWidget]) -> BaseModel:
    """Read current widget values back into a freshly-validated Params instance."""
    data: dict[str, Any] = {}
    for name, widget in widgets.items():
        if isinstance(widget, QComboBox):
            data[name] = widget.currentData()
        elif isinstance(widget, QLineEdit):
            data[name] = widget.text()
        else:
            data[name] = widget.value()
    return params_cls.model_validate(data)


def reload_params_form(params: BaseModel, widgets: dict[str, QWidget]) -> None:
    """Sync widget values from ``params`` (e.g. after Load settings)."""
    for name, widget in widgets.items():
        value = getattr(params, name)
        widget.blockSignals(True)
        if isinstance(widget, QComboBox):
            idx = widget.findData(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        else:
            widget.setValue(value)
        widget.blockSignals(False)
