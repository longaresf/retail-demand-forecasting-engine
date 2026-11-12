#!/usr/bin/env python3
"""
Script de automatización local para pruebas y despliegue de servicios.
Uso:
    python run_local_ops.py --project [forecasting] --action [test|run]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path) -> None:
    print(f"\n[EXEC] Running: {' '.join(command)} in {cwd}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatización de pruebas y despliegue local.")
    parser.add_argument(
        "--project",
        required=True,
        choices=["forecasting"],
        help="Proyecto sobre el cual ejecutar la operación.",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["test", "run"],
        help="Acción a realizar: 'test' (ejecutar pytest) o 'run' (levantar aplicación/pipeline).",
    )
    args = parser.parse_args()

    # Mapeo de rutas relativas (ajusta según la estructura de tu máquina)
    base_dirs = {
        "forecasting": Path("/home/fran/retail-demand-forecasting-engine"),
    }

    project_dir = base_dirs[args.project]
    venv_python = project_dir / ".venv" / "bin" / "python"
    venv_pytest = project_dir / ".venv" / "bin" / "pytest"
    venv_streamlit = project_dir / ".venv" / "bin" / "streamlit"

    if not venv_python.exists():
        print(f"[WARNING] No se encontró entorno virtual en {project_dir}. Creando .venv...")
        run_command(["python3", "-m", "venv", ".venv"], project_dir)
        run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], project_dir)
        run_command([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], project_dir)

    if args.action == "test":
        print(f"\n--- Ejecutando pruebas unitarias para {args.project} ---")
        run_command([str(venv_pytest), "-q"], project_dir)
        print("\n[SUCCESS] ¡Todas las pruebas pasaron exitosamente!")

    elif args.action == "run":
        if args.project == "forecasting":
            print("\n--- Iniciando dashboard de Streamlit (Forecasting) ---")
            run_command([str(venv_streamlit), "run", "app.py"], project_dir)


if __name__ == "__main__":
    main()