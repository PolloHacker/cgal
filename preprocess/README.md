# CGAL Point Cloud Preprocessing & Skeletonization

Este projeto fornece ferramentas em C++ baseadas na biblioteca [CGAL](https://www.cgal.org/) para processamento de nuvens de pontos 3D orientadas, reconstrução de superfícies estanques e extração de esqueletos topológicos por fluxo de curvatura média.

---

## Estrutura do Módulo

O diretório é dividido em dois executáveis principais:
1. **`preprocessor`**: Responsável pela carga, limpeza, suavização, subamostragem e estimativa estatística de espaçamento das nuvens de pontos.
2. **`skeletonization`**: Responsável pela normalização da malha triangular fechada, simplificação/decimação de faces e extração do esqueleto de curva.

---

## Etapas do Pipeline de Pré-processamento

1. **Leitura da nuvem de pontos com Subamostragem Espacial** (Opcional)
   - O pipeline carrega arquivos PLY contendo pontos 3D (ASCII ou Binário).
   - Se a **Subamostragem Espacial** estiver habilitada, os pontos são decimados por grade de voxels durante o streaming de leitura (carrega apenas um ponto representativo por voxel de tamanho `min_distance`).
   - Se o percentual de redução for $\ge 90\%$, executa-se automaticamente uma etapa de refinamento WLOP local para melhorar a regularidade dos pontos restantes.
   - Os atributos originais (como cores RGB, intensidade e normais) são mapeados de volta aos pontos regulares usando uma busca 1-NN em árvore KD-tree.

2. **Remoção de Outliers** (Opcional)
   - Remove pontos com base na distância média dos seus $k$ vizinhos mais próximos.

3. **WLOP Downsampling** (Opcional)
   - Aplica o algoritmo *Weighted Locally Optimal Projection* para simplificar e regularizar a distribuição dos pontos da nuvem de forma homogênea.

4. **Suavização por Jato (Jet Smoothing)** (Opcional)
   - Suaviza o ruído das coordenadas de pontos adaptando localmente superfícies polinomiais (jatos).

5. **Espaçamento Médio (Average Spacing)**
   - Calcula o espaçamento médio entre os pontos vizinhos, parâmetro crítico para ajustar a resolução e heurísticas da reconstrução de superfície subsequente.

---

## Etapas do Pipeline de Esqueletização

1. **Carga da Malha**
   - Carrega a malha triangular gerada pela reconstrução de Poisson.

2. **Normalização da Malha**
   - Garante que a malha cumpra as pré-condições matemáticas rigorosas para o fluxo de curvatura média (MCF):
     - Triangula faces não triangulares.
     - Isola e retém apenas o maior componente conectado.
     - Identifica e preenche furos (bordas abertas) por triangulação de Delaunay.
     - Costura bordas e junções abertas para garantir uma malha 100% estanque (fechada).

3. **Simplificação da Malha (Decimação)**
   - Caso a malha possua mais de 50.000 faces, ela é simplificada por colapso de arestas (CGAL Edge Collapse) até o limite de $\approx 75.000$ arestas para otimizar o tempo de CPU e consumo de memória da esqueletização.
   - Executa uma nova etapa de normalização pós-decimação para certificar a integridade do modelo.

4. **Extração de Esqueleto**
   - Roda o algoritmo de fluxo de curvatura média (Mean Curvature Flow) para contrair a malha mantendo sua topologia original.
   - Salva o esqueleto extraído em formato de polilinhas máximas (`.poly`), arestas em formato Wavefront OBJ (`_edges.obj`), e arquivo de correspondência vértice-esqueleto (`_correspondence.obj`).

---

## Como Usar

### Compilação (Build)

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Executar Pré-processamento (`preprocessor`)

```bash
./preprocessor <input_ply> <output_dir> [opções]
```

#### Opções Disponíveis

| Opção | Descrição | Padrão |
|---|---|---|
| `--enable-spatial-subsampling` | Ativa a subamostragem espacial por voxel durante a carga. | `false` |
| `--spatial-subsample-distance=VALUE` | Tamanho da aresta do voxel (também aceita `--min-distance=VALUE`). | `0.1` |
| `--remove-outliers-percent=VALUE` | Porcentagem de pontos ruidosos a remover. | `0.0` (desativado) |
| `--outlier-neighbors=K` | Número de vizinhos para análise de outliers e cálculo de espaçamento. | `6` |
| `--enable-smoothing` | Ativa a suavização por jato (jet smoothing). | `false` |
| `--smoothing-neighbors=K` | Quantidade de vizinhos usada na interpolação do jato. | `24` |
| `--enable-wlop` | Ativa a regularização espacial via WLOP. | `false` |
| `--wlop-retain-percent=VALUE` | Percentagem de pontos retida após simplificação WLOP. | `20.0` |
| `--wlop-neighbor-radius=VALUE` | Raio do vizinho WLOP (multiplicado pelo espaçamento médio se positivo). | `-1.0` (auto) |
| `--wlop-iterations=N` | Número de iterações do solver WLOP. | `35` |
| `--wlop-require-uniform-sampling` | Força amostragem estritamente uniforme no WLOP. | `false` |

### Executar Esqueletização (`skeletonization`)

```bash
./skeletonization <input_mesh.ply> <output_prefix>
```

#### Exemplo de Fluxo Completo

```bash
# 1. Pré-processar com subamostragem e remoção de ruídos
./preprocessor input/model.ply output_dir --enable-spatial-subsampling --min-distance=0.05 --remove-outliers-percent=1.0

# 2. Esqueletizar a malha gerada
./skeletonization output_dir/model/model_stage3_watertight.ply output_dir/model/skeleton_result
```
