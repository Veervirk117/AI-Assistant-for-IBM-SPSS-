# SPSS Statistical Analysis Assistant

A powerful RAG-powered bot that provides detailed explanations and step-by-step guidance for SPSS statistical analysis procedures.

## Bot Capabilities

This SPSS Assistant is a comprehensive statistical analysis helper that focuses on:

### 1. **Explainability** (Done)
- **Detailed Function Explanations**: Step-by-step instructions for every SPSS procedure
- **Menu Navigation**: Exact menu paths and clicks
- **Assumption Checking**: What to verify before running tests
- **Result Interpretation**: How to read and understand SPSS output
- **Common Pitfalls**: What users often get wrong and how to avoid it

### 2. **Doability** (Future Addition)
- **Action Suggestions**: What to do next based on results
- **Workflow Automation**: Automated execution of common procedures
- **Smart Recommendations**: Context-aware suggestions for analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SPSS Bot      │    │  Vector Store    │    │ Knowledge Base  │
│   (Main Logic)  │◄──►│  (RAG Engine)    │◄──►│  (SPSS Data)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Knowledge Coverage

The bot covers **8 major SPSS categories** with **24+ functions**:

### **Descriptive Statistics**
- Descriptive Statistics
- Frequency Tables

### **Data Management**
- Data Import
- Variable Recoding
- Missing Data Analysis

### **Inferential Statistics**
- Independent Samples T-Test
- One-Way ANOVA
- Correlation Analysis

### **Regression Analysis**
- Linear Regression
- Multiple Regression

### **Factor Analysis**
- Factor Analysis

### **Nonparametric Tests**
- Chi-Square Test
- Mann-Whitney U Test

### **Data Visualization**
- Histograms
- Scatter Plots

### **Output & Reporting**
- Output Management
- Syntax Editor

### Installation

1. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

### Running the Bot

#### **Interactive Mode** (Recommended)
```bash
python spss_bot.py
```

#### **Demo Mode** (Test functionality)
```bash
python demo_spss_bot.py
```

## How to Use

### **Basic Commands**
- `help` - Show all available SPSS functions
- `categories` - List all SPSS categories
- `difficulty` - Show functions by difficulty level
- `quit` - Exit the bot

### **Example Queries**

#### **Function-Specific Questions**
```
"How do I perform a t-test in SPSS?"
"What is factor analysis and how do I run it?"
"Show me step-by-step instructions for linear regression"
```

#### **Workflow Questions**
```
"How do I analyze survey data from start to finish?"
"What's the process for data cleaning and analysis?"
"Give me a workflow for hypothesis testing"
```

#### **Category Questions**
```
"What can I do with descriptive statistics?"
"Show me all regression analysis options"
"What visualization tools are available?"
```

## RAG Features

### **Smart Retrieval**
- **Semantic Search**: Finds relevant information even with different wording
- **Context-Aware**: Understands the user's intent and skill level
- **Filtered Results**: Can filter by category, difficulty, or function

### **Enhanced Context**
- **Related Functions**: Suggests complementary procedures
- **Workflow Planning**: Creates logical analysis sequences
- **Difficulty Matching**: Adapts explanations to user expertise

## File Structure

```
Newbot/
├── spss_bot.py              # Main bot interface
├── spss_knowledge_base.py   # SPSS knowledge database
├── spss_vector_store.py     # RAG vector store
├── demo_spss_bot.py         # Demo and testing script
├── requirements.txt          # Python dependencies
├── venv/                    # Virtual environment
└── README.md                # This file
```

This will test:
- Vector store initialization
- Knowledge base loading
- Query processing
- Response generation

## Customization

### **Adding New SPSS Functions**
Edit `spss_knowledge_base.py` and add new entries to `SPSS_KNOWLEDGE`:

```python
{
    "content": "Your function description...",
    "category": "Your Category",
    "function": "Function Name",
    "difficulty": "Beginner/Intermediate/Advanced",
    "steps": ["Step 1", "Step 2", "Step 3"]
}
```

### **Modifying Prompts**
Edit the prompt templates in `spss_bot.py` to change the bot's personality or response style.

### **Adjusting Search Parameters**
Modify `search_kwargs` in `spss_vector_store.py` to adjust retrieval behavior.

## Future Plans

### **Phase 2: Doability Features**
- **Action Execution**: Automatically perform SPSS procedures
- **Smart Workflows**: Context-aware analysis planning
- **Result Interpretation**: Automated insight generation
- **Data Validation**: Check assumptions automatically

### **Phase 3: Advanced Features**
- **Multi-language Support**: Support for different SPSS versions
- **Integration**: Connect with actual SPSS software
- **Learning**: Adapt to user preferences and common workflows

## Troubleshooting

### **Common Issues**

1. **"Model not found" error**:
   - Ensure Ollama is running: `ollama serve`
   - Pull the required model: `ollama pull llama3.2`

2. **"Embedding model not found"**:
   - Pull the embedding model: `ollama pull mxbai-embed-large`

3. **Vector store errors**:
   - Delete the `spss_knowledge_db` folder and restart
   - Check that all dependencies are installed

## 🎓 Use Cases

### **For Students**
- Learn SPSS step-by-step
- Understand statistical concepts
- Avoid common mistakes

### **For Researchers**
- Quick procedure lookups
- Workflow planning
- Best practice guidance

### **For Business Users**
- Data analysis procedures
- Report generation
- Statistical interpretation

---
