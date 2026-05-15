# CGAL Point Cloud Skeletonization

Este projeto converte nuvens de pontos orientadas em uma malha triangular fechada e extrai um esqueleto topo-geométrico usando as ferramentas CGAL.

## Etapas do pipeline

1. **Leitura da nuvem de pontos**
   - O pipeline carrega arquivos PLY contendo pontos 3D e, opcionalmente, normais.
   - Se as normais não estiverem disponíveis, os pontos são carregados sem normais e o pipeline pode estimá-las mais tarde.

2. **Pré-processamento do ponto**
   - Remove outliers com base em distância local e porcentagem configurável.
   - Opcionalmente, aplica WLOP para simplificação e regularização da nuvem de pontos.
   - Opcionalmente, aplica suavização por jato (jet smoothing).
   - Estima e orienta normais usando PCA + MST quando necessário.
   - Calcula o espaçamento médio dos pontos para ajustar parâmetros de reconstrução.

3. **Reconstrução de malha**
   - Gera uma malha triangular a partir dos pontos orientados usando a reconstrução de superfície de Poisson.
   - Usa parâmetros configuráveis de ângulo, raio e distância para controlar a qualidade da malha.

4. **Normalização de malha**
   - Triangula qualquer face não triangular.
   - Remove componentes desconectados menores, se configurado.
   - Preenche buracos de borda e costura fronteiras para garantir que a malha seja fechada.

5. **Extração do esqueleto**
   - Executa o algoritmo de fluxo de curvatura média sobre a malha fechada.
   - Produz um grafo esquelético com vértices e arestas embutidos na geometria.

6. **Exportação de resultados**
   - Salva o ponto bruto carregado e o ponto pré-processado como PLY.
   - Exporta a malha reconstruída antes e depois da normalização.
   - Salva o esqueleto como polilinhas máximas, lista de arestas e correspondência vértice-malha.

## Tecnologias usadas

- **CGAL**: algoritmo de reconstrução de superfície Poisson, estimativa de normais, remoção de outliers, suavização de pontos, WLOP e extração de esqueleto por fluxo de curvatura média.
- **Eigen3**: suporte a solvers esparsos necessários para a reconstrução de Poisson.
- **TBB**: paralelização de diversas etapas do pipeline.
- **CMake**: gerenciamento de build e dependências.

## Como usar

### Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Run

```bash
./skeletonization <input_ply> <output_dir> [opções]
```

### Exemplo

```bash
./skeletonization input/model.ply skeleton_output
```

### Opções principais

- `--remove-outliers-percent=VALUE`
- `--outlier-neighbors=K`
- `--normal-estimation-neighbors=K`
- `--normal-neighborhood-mode=fixed-k|spacing-radius`
- `--force-estimate-normals`
- `--keep-all-components`
- `--enable-smoothing`
- `--smoothing-neighbors=K`
- `--enable-wlop`
- `--wlop-retain-percent=VALUE`
- `--wlop-neighbor-radius=VALUE`
- `--wlop-iterations=N`
- `--wlop-require-uniform-sampling`
- `--sm_angle=VALUE`
- `--sm_radius=VALUE`
- `--sm_distance=VALUE`

> A execução cria um subdiretório em `output_dir` com o nome do arquivo de entrada e escreve os artefatos do pipeline nesse local.
