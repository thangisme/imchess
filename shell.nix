{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.ruff
    (pkgs.python3.withPackages (ps: [
      ps.scapy
      ps.jedi-language-server
      ps.python-lsp-server
      ps.chess
    ]))
    pkgs.pychess
  ];
}
