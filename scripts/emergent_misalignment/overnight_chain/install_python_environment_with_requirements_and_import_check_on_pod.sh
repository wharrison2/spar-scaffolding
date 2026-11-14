#!/usr/bin/env bash
# Creates one virtual environment that can still see the pod image's own packages (--system-site-packages), installs a requirements file
# into it and checks that the named modules import. Used twice by the overnight chain: the generation environment (vLLM brings its own
# torch, which must not replace the image's torch used by fine-tuning) and the fine-tuning environment (image torch plus peft).
#
#   bash install_python_environment_with_requirements_and_import_check_on_pod.sh <environment folder> <requirements file> <import check module> [...]
set -euo pipefail
environment_folder="$1"; requirements_file="$2"; shift 2
if [ ! -x "${environment_folder}/bin/python" ]; then
  python3 -m venv --system-site-packages "${environment_folder}" || {
    echo "python3 -m venv failed; trying virtualenv"
    python3 -m pip install -q virtualenv
    python3 -m virtualenv --system-site-packages "${environment_folder}"
  }
fi
"${environment_folder}/bin/python" -m pip install -q --upgrade pip
"${environment_folder}/bin/python" -m pip install -r "${requirements_file}"
for module in "$@"; do
  "${environment_folder}/bin/python" -c "import ${module}; print('${module}', getattr(${module}, '__version__', 'no version attribute'))"
done
"${environment_folder}/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"
# Steps call "${environment_folder}/bin/python" directly without activating the environment, so tools pip installed into its bin folder are
# not on PATH. flashinfer (vLLM's sampler) compiles kernels at engine start-up with the `ninja` program and fails with "No such file or
# directory: 'ninja'" when it cannot find it (first overnight run, 2026-11-14), so put the environment's ninja on the system PATH.
if [ -x "${environment_folder}/bin/ninja" ] && ! command -v ninja >/dev/null 2>&1; then
  ln -sf "${environment_folder}/bin/ninja" /usr/local/bin/ninja
  echo "linked ${environment_folder}/bin/ninja into /usr/local/bin"
fi
