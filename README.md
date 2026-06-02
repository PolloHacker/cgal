# Package for preprocessing point cloud and estimating normals

# wnnc

Este projeto utiliza o [uv](https://github.com) para gerenciamento rápido de ambientes virtuais e dependências Python, integrado com extensões C++/CUDA empacotadas via `setuptools`.

## Como Iniciar do Zero

Como o projeto possui dependências complexas de compilação C++ que importam o PyTorch diretamente no processo de build, a instalação deve ignorar o isolamento de build padrão do Python.

Siga os passos abaixo para criar o ambiente e instalar o projeto em modo editável:

```bash
# 1. Certifique-se de que possui o uv instalado
# (Caso não tenha: curl -sSf https://astral.sh | sh)

# 2. Sincronize e monte todo o ambiente virtual automaticamente
uv sync
cd wnnc/ext
uv run pip install -e . --no-build-isolation
```

O comando `uv sync` cuidará de criar a pasta `.venv`, baixar as versões exatas do PyTorch para a GPU e compilar a extensão local.


### Preprocess

Para compilar este programa, basta usar o `CMakeLists.txt` dentro da pasta `preprocess`. O binário estará na pasta `build` na raiz do projeto.

### AdaptiveSolvers

Esta pasta cuida da implementação do Poisson Reconstruction segundo [ KAZHDAN](https://github.com/mkazhdan/PoissonRecon). Para compilar e obter o binário `PoissonRecon`, faça:

```bash
sudo apt-get update
sudo apt-get install libjpeg-dev libturbojpeg0-dev ^Cib1g-dev libpng-dev

cd AdaptiveSolvers
mkdir JPEG
mkdir JPEG-turbo
mkdir PNG
mkdir ZLIB

make
```
Após algum tempo de compilação, um erro ocorrerá, mas o `PoissonRecon` terá sido compilado corretamente dentro de `Bin/Linux`.

---

## Detalhes de Configuração (`pyproject.toml`)

Se você precisar ajustar ou entender como o ambiente foi configurado, estas são as seções críticas do `pyproject.toml`:

### 1. Vínculo com CUDA 12.8 (Aceleração por Hardware)
Para evitar conflitos de versão entre o binário do PyTorch e o toolkit do sistema, o projeto aponta diretamente para o repositório oficial de wheels compiladas para **CUDA 12.8**:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://pytorch.org"
explicit = true  # Evita que outros pacotes busquem neste índice

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

### 2. Dependências de Build (`extra-build-dependencies`)
Alguns scripts internos de compilação exigem que o `torch` e o `numpy` estejam disponíveis imediatamente antes de iniciar o processo de empacotamento:

```toml
[tool.uv.extra-build-dependencies]
wnnc = ["torch", "numpy", "setuptools"]
```

---

## Ativando o Ambiente

Sempre que abrir um novo terminal e quiser rodar os scripts do projeto, ative o ambiente virtual gerado:

```bash
source .venv/bin/activate
```
