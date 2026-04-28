# 🎓 EduRisco — Sistema Inteligente de Previsão de Desempenho Escolar

> **Ferramenta de apoio à gestão pública municipal de educação**  
> Detecta precocemente alunos em risco de evasão ou baixo desempenho usando Inteligência Artificial.

---

## 📌 Impacto Social

A evasão escolar é um dos maiores desafios das redes públicas de ensino. Identificar alunos em situação de vulnerabilidade **antes** que o problema se agrave permite à gestão municipal:

- Direcionar recursos pedagógicos e psicossociais com precisão
- Reduzir custos de remediação e reforço emergencial
- Aumentar a taxa de conclusão do ensino médio no município
- Embasar decisões com dados, não apenas intuição

Este sistema processa **33 variáveis socioeconômicas e comportamentais** de cada aluno e retorna, em segundos, uma probabilidade de risco com os principais fatores que influenciaram o resultado — tornando a IA interpretável e acionável para gestores e professores.

---

## 🧠 Arquitetura do Modelo

| Componente | Detalhe |
|---|---|
| Algoritmo | Rede Neural MLP (Multilayer Perceptron) |
| Framework | TensorFlow / Keras |
| Camadas | Input → Dense(128, ReLU) → BN → Dropout(0.3) → Dense(64, ReLU) → BN → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid) |
| Otimizador | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Regularização | BatchNormalization + Dropout + EarlyStopping |
| Divisão | 80% treino / 20% teste (stratified) |
| Target | G3 < 10 → Alto Risco (classificação binária) |

---

## 📊 Dataset

**Student Alcohol Consumption** — UCI Machine Learning Repository / Kaggle

- **Origem:** Escolas secundárias de Portugal
- **Amostras:** ~395 alunos (matemática)
- **Features:** 33 variáveis (demográficas, familiares, escolares, comportamentais)
- **Variável-alvo criada:** `risco` (binária: G3 < 10)

### Principais variáveis utilizadas:
| Variável | Descrição |
|---|---|
| `failures` | Número de reprovações anteriores |
| `absences` | Total de faltas no ano |
| `studytime` | Horas de estudo semanais |
| `Dalc` / `Walc` | Consumo de álcool (dias úteis / fim de semana) |
| `famrel` | Qualidade das relações familiares |
| `Medu` / `Fedu` | Escolaridade dos pais |
| `higher` | Deseja cursar ensino superior |
| `goout` | Frequência de saídas sociais |
| `health` | Estado de saúde geral |

---

## 🚀 Como Rodar o Projeto

### Opção 1 — Streamlit Community Cloud (Recomendado para Deploy)

1. Faça fork deste repositório para sua conta GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub e selecione este repositório
4. Defina `app.py` como arquivo principal
5. Clique em **Deploy** — URL pública gerada em ~2 minutos ✅

### Opção 2 — Rodar Localmente

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/edurisco.git
cd edurisco

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Execute o app
streamlit run app.py
```

O dataset é carregado automaticamente via URL pública. Nenhuma configuração adicional necessária.

---

## 📁 Estrutura de Arquivos

```
edurisco/
├── app.py               # Código principal: pipeline de dados + modelo + interface Streamlit
├── requirements.txt     # Dependências para deploy
└── README.md            # Esta documentação
```

> **Nota sobre persistência do modelo:** Em produção com múltiplas escolas, salve o `StandardScaler` e o modelo treinado usando `joblib` e `modelo.save()` para garantir consistência na normalização de novos dados:
> ```python
> joblib.dump(scaler, "scaler.pkl")
> modelo.save("modelo.keras")
> ```

---

## ⚖️ Diretrizes Éticas e LGPD

Este sistema foi construído com responsabilidade ética em primeiro lugar:

1. **Ferramenta de apoio, nunca de julgamento:** A previsão de "Alto Risco" deve iniciar um processo de acolhimento, nunca de exclusão ou punição do aluno.

2. **Dados sensíveis tratados com responsabilidade:** Variáveis como consumo de álcool são interpretadas como **indicadores de vulnerabilidade social** — não como falhas de caráter. O acesso ao sistema deve ser restrito a profissionais autorizados.

3. **LGPD (Lei 13.709/2018):** A implementação em ambiente municipal exige: base legal para tratamento dos dados, minimização de dados coletados, controle de acesso por perfil e política de retenção definida. Recomenda-se consultar o DPO (Encarregado de Dados) da prefeitura antes do deploy em produção.

4. **Viés algorítmico:** O modelo foi treinado em dados de escolas portuguesas. Antes do uso em larga escala, recomenda-se retreinar com dados locais e auditar os resultados para garantir equidade entre grupos.

5. **Revisão humana:** Nenhuma decisão de intervenção deve ser tomada apenas com base na previsão do sistema. A análise pedagógica por profissionais qualificados é indispensável.

---

## 👥 Equipe

Desenvolvido como projeto aplicado de Machine Learning para gestão pública municipal.  
**Equipe:** Marcos Santos, Marcelo Augusto, Thiago vasconcelos, Diego veloso, Felipe Sousa

---

## 📜 Licença

MIT License — uso livre para fins educacionais e de gestão pública. Atribuição recomendada.
