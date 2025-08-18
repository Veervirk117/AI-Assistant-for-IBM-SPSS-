from langchain_core.documents import Document
import json

# Comprehensive SPSS Knowledge Base
SPSS_KNOWLEDGE = [
    # Data Management
    {
        "content": "SPSS Data Import: To import data, go to File > Open > Data. Supported formats include .sav, .csv, .xlsx, .txt. For CSV files, ensure proper delimiter settings and variable names in first row.",
        "category": "Data Management",
        "function": "Data Import",
        "difficulty": "Beginner",
        "steps": [
            "File > Open > Data",
            "Select file format (.sav, .csv, .xlsx, .txt)",
            "For CSV: Set delimiter and variable names",
            "Click OK to import"
        ]
    },
    {
        "content": "Variable Recoding: Use Transform > Recode > Into Different Variables to create new variables. Specify old and new values, handle missing data, and apply labels for clarity.",
        "category": "Data Management",
        "function": "Variable Recoding",
        "difficulty": "Intermediate",
        "steps": [
            "Transform > Recode > Into Different Variables",
            "Select source variable",
            "Define old and new values",
            "Handle missing data appropriately",
            "Add variable labels"
        ]
    },
    {
        "content": "Missing Data Analysis: Use Analyze > Missing Value Analysis to identify patterns. Options include listwise deletion, pairwise deletion, or multiple imputation for handling missing values.",
        "category": "Data Management",
        "function": "Missing Data Analysis",
        "difficulty": "Advanced",
        "steps": [
            "Analyze > Missing Value Analysis",
            "Select variables for analysis",
            "Choose missing data handling method",
            "Review patterns and decide on approach"
        ]
    },
    
    # Descriptive Statistics
    {
        "content": "Descriptive Statistics: Use Analyze > Descriptive Statistics > Descriptives to get mean, standard deviation, minimum, maximum, and other summary statistics for continuous variables.",
        "category": "Descriptive Statistics",
        "function": "Descriptive Statistics",
        "difficulty": "Beginner",
        "steps": [
            "Analyze > Descriptive Statistics > Descriptives",
            "Select variables for analysis",
            "Choose statistics (mean, std dev, min, max, etc.)",
            "Click OK to run analysis"
        ]
    },
    {
        "content": "Frequency Tables: Use Analyze > Descriptive Statistics > Frequencies to create frequency distributions for categorical variables. Include percentages, cumulative percentages, and charts.",
        "category": "Descriptive Statistics",
        "function": "Frequency Tables",
        "difficulty": "Beginner",
        "steps": [
            "Analyze > Descriptive Statistics > Frequencies",
            "Select categorical variables",
            "Choose statistics (frequencies, percentages, charts)",
            "Click OK to generate tables"
        ]
    },
    
    # Inferential Statistics
    {
        "content": "T-Test (Independent Samples): Use Analyze > Compare Means > Independent-Samples T Test to compare means between two groups. Ensure variables meet assumptions of normality and equal variances.",
        "category": "Inferential Statistics",
        "function": "Independent Samples T-Test",
        "difficulty": "Intermediate",
        "steps": [
            "Analyze > Compare Means > Independent-Samples T Test",
            "Select test variable (dependent variable)",
            "Select grouping variable (independent variable)",
            "Define groups (e.g., 1 and 2)",
            "Click OK and interpret results"
        ]
    },
    {
        "content": "One-Way ANOVA: Use Analyze > Compare Means > One-Way ANOVA to compare means across three or more groups. Check homogeneity of variances and normality assumptions.",
        "category": "Inferential Statistics",
        "function": "One-Way ANOVA",
        "difficulty": "Intermediate",
        "steps": [
            "Analyze > Compare Means > One-Way ANOVA",
            "Select dependent variable",
            "Select factor (independent variable)",
            "Set confidence level (default 95%)",
            "Click OK and check F-statistic and p-value"
        ]
    },
    {
        "content": "Correlation Analysis: Use Analyze > Correlate > Bivariate to examine relationships between variables. Choose correlation coefficient (Pearson, Spearman) based on data type and distribution.",
        "category": "Inferential Statistics",
        "function": "Correlation Analysis",
        "difficulty": "Intermediate",
        "steps": [
            "Analyze > Correlate > Bivariate",
            "Select variables for correlation",
            "Choose correlation coefficient",
            "Set significance level",
            "Click OK and interpret correlation matrix"
        ]
    },
    
    # Regression Analysis
    {
        "content": "Linear Regression: Use Analyze > Regression > Linear to predict dependent variable from independent variables. Check assumptions: linearity, normality, homoscedasticity, and independence.",
        "category": "Regression Analysis",
        "function": "Linear Regression",
        "difficulty": "Advanced",
        "steps": [
            "Analyze > Regression > Linear",
            "Select dependent variable",
            "Select independent variables",
            "Choose method (Enter, Stepwise, etc.)",
            "Check assumptions and interpret results"
        ]
    },
    {
        "content": "Multiple Regression: Use Analyze > Regression > Linear for multiple predictors. Consider multicollinearity, use stepwise selection if needed, and validate model assumptions.",
        "category": "Regression Analysis",
        "function": "Multiple Regression",
        "difficulty": "Advanced",
        "steps": [
            "Analyze > Regression > Linear",
            "Select dependent variable",
            "Add multiple independent variables",
            "Check for multicollinearity (VIF)",
            "Validate assumptions and interpret model"
        ]
    },
    
    # Factor Analysis
    {
        "content": "Factor Analysis: Use Analyze > Dimension Reduction > Factor to identify underlying latent variables. Check KMO, Bartlett's test, and use appropriate rotation methods.",
        "category": "Factor Analysis",
        "function": "Factor Analysis",
        "difficulty": "Advanced",
        "steps": [
            "Analyze > Dimension Reduction > Factor",
            "Select variables for analysis",
            "Choose extraction method (Principal Components)",
            "Select rotation method (Varimax, Oblimin)",
            "Interpret factor loadings and communalities"
        ]
    },
    
    # Nonparametric Tests
    {
        "content": "Chi-Square Test: Use Analyze > Nonparametric Tests > Chi-Square to test independence between categorical variables. Ensure expected frequencies are adequate (>5).",
        "category": "Nonparametric Tests",
        "function": "Chi-Square Test",
        "difficulty": "Intermediate",
        "steps": [
            "Analyze > Nonparametric Tests > Chi-Square",
            "Select test variables",
            "Check expected frequencies",
            "Click OK and interpret chi-square statistic",
            "Check p-value for significance"
        ]
    },
    {
        "content": "Mann-Whitney U Test: Use Analyze > Nonparametric Tests > Independent Samples to compare two groups when data doesn't meet t-test assumptions.",
        "category": "Nonparametric Tests",
        "function": "Mann-Whitney U Test",
        "difficulty": "Intermediate",
        "steps": [
            "Analyze > Nonparametric Tests > Independent Samples",
            "Select test variables",
            "Define groups",
            "Choose Mann-Whitney U test",
            "Interpret U statistic and p-value"
        ]
    },
    
    # Data Visualization
    {
        "content": "Histograms: Use Graphs > Legacy Dialogs > Histogram to visualize distribution of continuous variables. Check for normality, skewness, and outliers.",
        "category": "Data Visualization",
        "function": "Histograms",
        "difficulty": "Beginner",
        "steps": [
            "Graphs > Legacy Dialogs > Histogram",
            "Select variable for histogram",
            "Choose display options (normal curve, etc.)",
            "Click OK to generate chart"
        ]
    },
    {
        "content": "Scatter Plots: Use Graphs > Legacy Dialogs > Scatter/Dot to examine relationships between two continuous variables. Add regression line if appropriate.",
        "category": "Data Visualization",
        "function": "Scatter Plots",
        "difficulty": "Beginner",
        "steps": [
            "Graphs > Legacy Dialogs > Scatter/Dot",
            "Select X and Y variables",
            "Choose scatter plot type",
            "Add regression line if needed",
            "Click OK to generate plot"
        ]
    },
    
    # Output and Reporting
    {
        "content": "SPSS Output: Results appear in Output Viewer. Right-click to copy, export as Word/Excel, or save as .spv file. Use Edit > Options to customize output format.",
        "category": "Output and Reporting",
        "function": "Output Management",
        "difficulty": "Beginner",
        "steps": [
            "View results in Output Viewer",
            "Right-click to copy or export",
            "File > Export to save in different formats",
            "Edit > Options to customize display"
        ]
    },
    {
        "content": "Syntax Editor: Use File > New > Syntax to write and save SPSS commands. Syntax provides reproducibility and automation for complex analyses.",
        "category": "Output and Reporting",
        "function": "Syntax Editor",
        "difficulty": "Advanced",
        "steps": [
            "File > New > Syntax",
            "Write SPSS commands",
            "Use Run > All to execute",
            "Save .sps file for future use"
        ]
    }
]

def create_spss_documents():
    """Convert SPSS knowledge base to Document objects for vector storage"""
    documents = []
    
    for i, item in enumerate(SPSS_KNOWLEDGE):
        # Create detailed content with metadata
        content = f"""
Function: {item['function']}
Category: {item['category']}
Difficulty: {item['difficulty']}

{item['content']}

Step-by-Step Process:
{chr(10).join([f"{j+1}. {step}" for j, step in enumerate(item['steps'])])}
"""
        
        doc = Document(
            page_content=content,
            metadata={
                "function": item['function'],
                "category": item['category'],
                "difficulty": item['difficulty'],
                "steps": " | ".join(item['steps']),  # Convert list to string
                "id": str(i)
            }
        )
        documents.append(doc)
    
    return documents

def get_spss_categories():
    """Get list of available SPSS categories"""
    return list(set([item['category'] for item in SPSS_KNOWLEDGE]))

def get_spss_functions():
    """Get list of available SPSS functions"""
    return [item['function'] for item in SPSS_KNOWLEDGE]

def get_spss_by_difficulty(difficulty):
    """Get SPSS functions by difficulty level"""
    return [item for item in SPSS_KNOWLEDGE if item['difficulty'].lower() == difficulty.lower()] 