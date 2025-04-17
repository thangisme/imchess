{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.ruff
    pkgs.vscode-langservers-extracted
    pkgs.superhtml
    pkgs.typescript-language-server
    (pkgs.python3.withPackages (ps: [
      ps.scapy
      ps.jedi-language-server
      ps.python-lsp-server
      ps.chess
      ps.fastapi
      ps.uvicorn
      ps.websockets
      ps.jinja2
      ps.tensorflow
      ps.keras
      ps.zstandard
      ps.numpy
      ps.matplotlib
      ps.orjson
    ]))
    pkgs.pychess
  ];
}
