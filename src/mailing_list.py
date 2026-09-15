# -*- coding: utf-8 -*-
r"""
Opt-in dialog for the IfE software mailing list.

Shared by every program in the PCC-Suite: PCC-Explorer, PCC-Viewer,
ATX-Converter, ATX-Scanner, RINEX-Masker and RINEX-Adapter. The PCC-Suite
launcher deliberately does not show it, because it only starts the others.

The list exists to reach users when something changes that would otherwise
break their work, for example the AIUB orbit server moving to a new address.
Subscribing is voluntary and stays the user's choice. The list is moderated:
only the maintainers can post, and every subscription is approved.

Two subscription routes are offered. The web form is the main one, because a
mailto link only does something on a machine that has a desktop mail program
configured, and a webmail user would see nothing happen at all.

Every program of the suite uses this same module, unchanged, so the dialog and
the stored answer behave identically everywhere.

The answer is stored once per machine, in
    %APPDATA%\PCC-Suite\mailing_list.json
so that a user who answers in one program is not asked again by the other five.
The file is separate from the launcher's settings.json on purpose: two programs
running at once cannot then overwrite each other's data.

No module-level side effects, and nothing here may take a program down: every
public function swallows its own errors. Console output stays ASCII, because a
Windows console is cp1252 and a print must never be able to change control flow.
"""

import json
import os
import sys
import webbrowser
from datetime import datetime, timedelta
from urllib.parse import quote

import tkinter as tk
from tkinter import messagebox

try:
    import ttkbootstrap as ttk
except Exception:                                    # plain tkinter fallback
    from tkinter import ttk


# --- The institute software mailing list ------------------------------------
LIST_NAME = "SOFTWARE-IFE"
LIST_ADDRESS = "SOFTWARE-IFE@LISTSERV.UNI-HANNOVER.DE"
LISTSERV_ADDRESS = "listserv@listserv.uni-hannover.de"
SUBSCRIBE_COMMAND = "subscribe software-ife"
SUBSCRIBE_URL = (
    "https://listserv.uni-hannover.de/cgi-bin/wa?SUBED1=SOFTWARE-IFE&A=1"
)

# How long "No thanks" keeps the dialog away. "Do not show this again" is final.
REMIND_AFTER_DAYS = 30

_STATE_VERSION = 1
_WRAP = 520


# --- Stored answer ----------------------------------------------------------

def state_path():
    r"""%APPDATA%\PCC-Suite\mailing_list.json, or a dot folder off $HOME."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "PCC-Suite", "mailing_list.json")
    return os.path.join(os.path.expanduser("~"), ".pcc-suite", "mailing_list.json")


def load_state():
    try:
        with open(state_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(data):
    try:
        path = state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return True
    except Exception as exc:
        print("[WARNING] Could not store the mailing list answer: %s" % exc)
        return False


def _record(choice, program_name):
    state = load_state()
    state.update({
        "version": _STATE_VERSION,
        "choice": choice,
        "answered_in": program_name,
        "answered_on": datetime.now().strftime("%Y-%m-%d"),
        "list": LIST_ADDRESS,
    })
    save_state(state)


def should_ask():
    """True when the dialog still has to be shown on this machine."""
    try:
        state = load_state()
        choice = state.get("choice")
        if choice in ("subscribed", "never"):
            return False
        if choice == "later":
            try:
                last = datetime.strptime(state.get("answered_on", ""), "%Y-%m-%d")
            except Exception:
                return True
            return datetime.now() - last >= timedelta(days=REMIND_AFTER_DAYS)
        return True
    except Exception:
        return False


def answered_summary():
    """One short line about the stored answer, or an empty string."""
    state = load_state()
    choice = state.get("choice")
    when = state.get("answered_on", "")
    where = state.get("answered_in", "")
    if not choice:
        return ""
    words = {
        "subscribed": "You opened the subscription form",
        "later": "You chose No thanks",
        "never": "You asked not to be reminded again",
    }
    text = words.get(choice, "You answered")
    if when:
        text += " on %s" % when
    if where:
        text += " in %s" % where
    return text + "."


# --- Small helpers ----------------------------------------------------------

def _open_url(url):
    try:
        webbrowser.open(url)
        return True
    except Exception as exc:
        print("[WARNING] Could not open %s (%s)" % (url, exc))
        return False


def mailto_url():
    """A mailto whose body is the command line and nothing else."""
    return "mailto:%s?subject=&body=%s" % (
        LISTSERV_ADDRESS, quote(SUBSCRIBE_COMMAND),
    )


def _button(master, text, command, style=None, width=None):
    """ttkbootstrap button that still builds under plain tkinter.ttk."""
    kwargs = {"text": text, "command": command}
    if width:
        kwargs["width"] = width
    if style:
        try:
            return ttk.Button(master, bootstyle=style, **kwargs)
        except Exception:
            pass
    return ttk.Button(master, **kwargs)


def _copy(widget, text):
    try:
        widget.clipboard_clear()
        widget.clipboard_append(text)
        widget.update_idletasks()
    except Exception:
        pass


def _copy_row(master, caption, value, width=34):
    """Caption, a read-only but selectable field, and a Copy button.

    The text is inserted into the widget rather than bound to a StringVar. A
    variable created here would be the only reference to itself, so Python
    would collect it and the field would render empty, which is what the first
    rendered test of this dialog showed.
    """
    row = ttk.Frame(master)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=caption, width=13).pack(side="left")
    entry = ttk.Entry(row, width=width)
    entry.pack(side="left", fill="x", expand=True)
    entry.insert(0, value)
    entry.configure(state="readonly")
    _button(row, "Copy", lambda: _copy(row, value), style="secondary",
            width=6).pack(side="left", padx=(6, 0))
    return row


def _centre(window, parent):
    try:
        window.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        x = max(0, pw - window.winfo_width() // 2)
        y = max(0, ph - window.winfo_height() // 2)
        window.geometry("+%d+%d" % (x, y))
    except Exception:
        pass


# --- The dialog -------------------------------------------------------------

def show_dialog(parent, program_name="PCC-Suite"):
    """
    Show the opt-in dialog and store whatever the user chooses.

    Four choices: subscribe through the web form, subscribe by
    e-mail instead, no thanks, or never show this again.
    """
    dialog = tk.Toplevel(parent)
    dialog.title("IfE software mailing list")
    dialog.resizable(False, False)
    dialog.transient(parent)

    frame = ttk.Frame(dialog, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Stay informed about IfE software updates",
              font=("Helvetica", 12, "bold")).pack(anchor="w")

    ttk.Label(
        frame,
        text=("The Institut für Erdmessung runs a mailing list for its GNSS "
              "software. It announces new releases and warns you about changes "
              "that can break your work, for example when a server for "
              "satellite orbit products moves to a new address."),
        wraplength=_WRAP, justify="left",
    ).pack(anchor="w", pady=(10, 0))

    ttk.Label(
        frame,
        text=("It is moderated, so only the maintainers can post and traffic "
              "is low. Subscribing is voluntary, %s works without it, and you "
              "can leave at any time." % program_name),
        wraplength=_WRAP, justify="left",
    ).pack(anchor="w", pady=(6, 0))

    ttk.Separator(frame).pack(fill="x", pady=12)

    ttk.Label(
        frame,
        text=("Subscribe opens the subscription form in your browser. To use "
              "e-mail instead, send the command line below to the LISTSERV "
              "address, on its own: LISTSERV reads the body line by line, so a "
              "signature from your mail program can stop it."),
        wraplength=_WRAP, justify="left",
    ).pack(anchor="w", pady=(0, 8))

    _copy_row(frame, "List address:", LIST_ADDRESS)
    _copy_row(frame, "Send to:", LISTSERV_ADDRESS)
    _copy_row(frame, "Message body:", SUBSCRIBE_COMMAND)

    ttk.Label(
        frame,
        text=("LISTSERV answers with a confirmation mail. Your subscription "
              "becomes active only after you reply to it and a moderator "
              "approves the request."),
        wraplength=_WRAP, justify="left", foreground="#ffc107",
    ).pack(anchor="w", pady=(12, 0))

    summary = answered_summary()
    if summary:
        ttk.Label(frame, text=summary, wraplength=_WRAP, justify="left",
                  foreground="#8b949e").pack(anchor="w", pady=(10, 0))

    # --- choices ---
    def _finish(choice):
        _record(choice, program_name)
        try:
            dialog.grab_release()
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    def _after_subscribe(route):
        messagebox.showinfo(
            "One more step",
            "%s\n\n"
            "LISTSERV will send you a confirmation mail. Please reply to it, "
            "otherwise the subscription is not completed. A moderator then "
            "approves the request.\n\n"
            "List: %s" % (route, LIST_ADDRESS),
            parent=parent,
        )

    def _subscribe_web():
        opened = _open_url(SUBSCRIBE_URL)
        _finish("subscribed")
        if opened:
            _after_subscribe("The subscription form was opened in your browser.")
        else:
            messagebox.showwarning(
                "Could not open the browser",
                "Please open this address by hand:\n\n%s" % SUBSCRIBE_URL,
                parent=parent,
            )

    def _subscribe_mail():
        opened = _open_url(mailto_url())
        _finish("subscribed")
        if opened:
            _after_subscribe(
                "A new mail to %s was prepared, with the single line "
                "'%s' in the body. Please send it without adding anything "
                "else." % (LISTSERV_ADDRESS, SUBSCRIBE_COMMAND))
        else:
            messagebox.showwarning(
                "Could not open your mail program",
                "Please send a mail to %s with the single line\n\n%s\n\n"
                "in the message body." % (LISTSERV_ADDRESS, SUBSCRIBE_COMMAND),
                parent=parent,
            )

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(18, 0))

    _button(buttons, "Subscribe", _subscribe_web,
            style="success").pack(side="left")
    _button(buttons, "Subscribe by e-mail instead", _subscribe_mail,
            style="info-outline").pack(side="left", padx=(8, 0))
    _button(buttons, "No thanks", lambda: _finish("later"),
            style="secondary").pack(side="right")
    # A plain link, so it does not compete with the other three and does not
    # read as a disabled button, which the outline style did.
    _button(buttons, "Do not show this again", lambda: _finish("never"),
            style="link").pack(side="right", padx=(0, 4))

    dialog.protocol("WM_DELETE_WINDOW", lambda: _finish("later"))
    dialog.bind("<Escape>", lambda e: _finish("later"))

    _centre(dialog, parent)
    try:
        dialog.wait_visibility()
        dialog.grab_set()
    except Exception:
        pass
    return dialog


# --- What the programs call -------------------------------------------------

def maybe_show(parent, program_name, delay_ms=900):
    """
    Show the dialog shortly after start-up, unless the user already answered.

    The delay lets the main window paint first, so the program never looks as
    if it opened with a pop-up and nothing behind it.
    """
    try:
        if not should_ask():
            return False
        parent.after(delay_ms, lambda: _safe_show(parent, program_name))
        return True
    except Exception as exc:
        print("[WARNING] Mailing list dialog not scheduled: %s" % exc)
        return False


def _safe_show(parent, program_name):
    try:
        if parent.winfo_exists():
            show_dialog(parent, program_name)
    except Exception as exc:
        print("[WARNING] Mailing list dialog failed: %s" % exc)


def show_on_request(parent, program_name):
    """The Mailing List button: always shows, whatever was answered before."""
    try:
        show_dialog(parent, program_name)
    except Exception as exc:
        print("[WARNING] Mailing list dialog failed: %s" % exc)
