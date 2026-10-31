"""Local static server for this folder: py web/serve.py [port]

Same as `py -m http.server -d web`, but forces the right MIME type for .js/.mjs. On Windows the stock
server can answer text/plain (from a registry entry) and browsers then refuse to load ES modules.
GitHub Pages serves the right types by itself."""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css"}


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
root = os.path.dirname(os.path.abspath(__file__))
print(f"serving {root} at http://127.0.0.1:{port}/", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=root)).serve_forever()
