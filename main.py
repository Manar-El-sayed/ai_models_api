#!/usr/bin/env python
# coding: utf-8

# In[5]:


import logging
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
import joblib
from typing import Dict, Any, List
import uvicorn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model paths 

DIABETES_MODEL_PATH = Path("models/diabetes.pkl")
DIABETES_SCALER_PATH = Path("models/scalerdiabetes.pkl")
ANEMIA_MODEL_PATH = Path("models/Ml_Anemia_graduation.pkl")
ANEMIA_SCALER_PATH = Path("models/ScalerAnemia.pkl")
HEPC_MODEL_PATH = Path("models/RF VC.pkl")
HEPC_SCALER_PATH = Path("models/scalerRF VC.pkl")
KIDNEY_MODEL_PATH = Path("models/Chronickidney.pkl")
KIDNEY_SCALER_PATH = Path("models/ScalerChronickidney.pkl")
HEART_MODEL_PATH = Path("models/heart_disease_model.pkl")
HEART_SCALER_PATH = Path("models/scaler_heart.pkl")
CANSER_MODEL_PATH = Path("models/canser.pkl")
CANSER_SCALER_PATH = Path("models/canserscaler.pkl")

# -------------------- Diabetes Models --------------------

class DiabetesInput(BaseModel):
    age: int = Field(..., description="Age of the patient")
    hypertension: int = Field(..., description="Hypertension (0 for No, 1 for Yes)")
    bmi: float = Field(..., description="Body Mass Index")
    hbA1c_level: float = Field(..., description="HbA1c Level")
    blood_glucose_level: float = Field(..., description="Blood Glucose Level")

@model_validator(mode='after')
def validate_fields(self) -> 'DiabetesInput':
    # Validate age
    if self.age < 0 or self.age > 120:
        raise ValueError('Age must be between 0 and 120 years')

    # Validate hypertension
    if self.hypertension not in [0, 1]:
        raise ValueError('Hypertension must be 0 (No) or 1 (Yes)')

    # Validate BMI
    if self.bmi <= 0 or self.bmi > 100:
        raise ValueError('BMI must be between 0 and 100')

    # Validate HbA1c
    if self.hbA1c_level <= 0 or self.hbA1c_level > 20:
        raise ValueError('HbA1c level must be between 0 and 20')

    # Validate blood glucose
    if self.blood_glucose_level <= 0:
        raise ValueError('Blood glucose level must be positive')

    return self

model_config = {
    "json_schema_extra": {
        "example": {
            "age": 45,
            "hypertension": 0,
            "bmi": 28.5,
            "hbA1c_level": 6.5,
            "blood_glucose_level": 140
        }
    }
}

class DiabetesPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendation: str
    input_values: Dict[str, Any]

def get_diabetes_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def get_diabetes_recommendation(risk_level: str, age: int, bmi: float) -> str:
    recommendations = {
        "Low Risk": (
            "Your diabetes risk appears to be low. To maintain this: \n"
            "- Continue maintaining a healthy lifestyle\n"
            "- Regular exercise for 30 minutes daily\n"
            "- Maintain a balanced diet"
        ),
        "Moderate Risk": (
            "You have moderate risk factors for diabetes: \n"
            "- Schedule a consultation with your healthcare provider\n"
            "- Consider lifestyle modifications\n"
            "- Monitor your blood sugar regularly"
        ),
        "High Risk": (
            "You show high risk factors for diabetes: \n"
            "- Immediate consultation with a healthcare provider is recommended\n"
            "- Comprehensive diabetes screening may be needed\n"
            "- Start monitoring blood sugar levels closely"
        )
    }
    
    base_recommendation = recommendations.get(risk_level, "Please consult a healthcare provider for proper evaluation.")
    
    # Add BMI-specific advice
    if bmi >= 30:
        base_recommendation += "\nConsider weight management strategies as your BMI indicates obesity."
    elif bmi >= 25:
        base_recommendation += "\nConsider weight management strategies as your BMI indicates overweight."
    
    return base_recommendation
# -------------------- Anemia Models --------------------

class AnemiaInput(BaseModel):
    hemoglobin: float = Field(..., description="Hemoglobin level (g/dL)")
    mch: float = Field(..., description="Mean Corpuscular Hemoglobin (pg)")
    mchc: float = Field(..., description="Mean Corpuscular Hemoglobin Concentration (g/dL)")
    mcv: float = Field(..., description="Mean Corpuscular Volume (fL)")
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'AnemiaInput':
            
        for field_name, value in {
            'hemoglobin': self.hemoglobin,
            'mch': self.mch,
            'mchc': self.mchc,
            'mcv': self.mcv
        }.items():
            if value <= 0:
                raise ValueError(f'{field_name} must be positive')
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "hemoglobin": 11.5,
                "mch": 28.0,
                "mchc": 33.0,
                "mcv": 85.0
            }
        }
    }

class DetailedAnalysis(BaseModel):
    parameter: str
    value: float
    status: str
    interpretation: str
    recommendations: List[str]

class AnemiaPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]

class ParameterRanges:
    def __init__(self):
        self.ranges = {
            'hemoglobin': {'low': 12.0, 'high': 17.5},  
            'mch': {'low': 27.0, 'high': 32.0},
            'mchc': {'low': 32.0, 'high': 36.0},
            'mcv': {'low': 80.0, 'high': 100.0}
        }

    def get_range(self, parameter: str) -> Dict[str, float]:
        return self.ranges[parameter]



def analyze_parameter(param_name: str, value: float, ranges: ParameterRanges) -> DetailedAnalysis:
    param_range = ranges.get_range(param_name)
    status = "Normal"
    interpretation = ""
    recommendations = []

    if value < param_range['low']:
        status = "Low"
        if param_name == 'hemoglobin':
            interpretation = "Low hemoglobin indicates reduced oxygen-carrying capacity in the blood."
            recommendations = [
                "Increase iron-rich foods in your diet (red meat, leafy greens, beans)",
                "Consider vitamin C supplements to enhance iron absorption",
                "Get tested for iron deficiency",
                "Discuss iron supplementation with your healthcare provider"
            ]
        elif param_name == 'mch':
            interpretation = "Low MCH suggests possible iron deficiency or chronic disease."
            recommendations = [
                "Include more iron-rich foods in your diet",
                "Consider B12 and folate supplementation",
                "Get tested for nutritional deficiencies"
            ]
        elif param_name == 'mchc':
            interpretation = "Low MCHC might indicate iron deficiency or thalassemia."
            recommendations = [
                "Seek further testing for iron deficiency",
                "Consider genetic testing for thalassemia",
                "Consult a hematologist"
            ]
        elif param_name == 'mcv':
            interpretation = "Low MCV indicates microcytic anemia, commonly due to iron deficiency."
            recommendations = [
                "Increase iron intake through diet or supplements",
                "Get tested for iron deficiency and lead exposure",
                "Consider testing for thalassemia"
            ]
    elif value > param_range['high']:
        status = "High"
        if param_name == 'hemoglobin':
            interpretation = "Elevated hemoglobin might indicate polycythemia or dehydration."
            recommendations = [
                "Increase water intake",
                "Avoid smoking and altitude exposure",
                "Get tested for polycythemia vera",
                "Consider sleep study for sleep apnea"
            ]
        elif param_name == 'mch':
            interpretation = "High MCH often indicates macrocytic anemia."
            recommendations = [
                "Get tested for vitamin B12 and folate levels",
                "Consider liver function tests",
                "Discuss alcohol consumption with healthcare provider"
            ]
        elif param_name == 'mchc':
            interpretation = "Elevated MCHC might indicate spherocytosis or lab error."
            recommendations = [
                "Verify results with repeat testing",
                "Consider hereditary spherocytosis testing",
                "Consult a hematologist"
            ]
        elif param_name == 'mcv':
            interpretation = "High MCV suggests macrocytic anemia, often due to B12 or folate deficiency."
            recommendations = [
                "Get tested for vitamin B12 and folate deficiency",
                "Evaluate alcohol consumption",
                "Consider thyroid function tests"
            ]
    else:
        interpretation = f"{param_name.upper()} is within normal range."
        recommendations = ["Continue current dietary and lifestyle habits"]

    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_anemia_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_overall_recommendation(detailed_analyses: List[DetailedAnalysis], risk_level: str) -> str:
    abnormal_params = [analysis for analysis in detailed_analyses if analysis.status != "Normal"]
    
    if not abnormal_params:
        return ("Your blood parameters are within normal ranges. Continue maintaining a healthy diet "
                "and lifestyle. Regular check-ups are recommended for monitoring.")
    
    recommendation_parts = []
    
    if risk_level == "High Risk":
        recommendation_parts.append("URGENT: Schedule an appointment with a healthcare provider for comprehensive evaluation.")
    
    if any(analysis.parameter == "hemoglobin" and analysis.status == "Low" for analysis in abnormal_params):
        recommendation_parts.append(
            "Priority: Address low hemoglobin levels through dietary changes and possible supplementation. "
            "Include iron-rich foods and vitamin C to enhance absorption."
        )
    
    nutritional_issues = any(analysis.status == "Low" for analysis in abnormal_params)
    if nutritional_issues:
        recommendation_parts.append(
            "Consider comprehensive nutritional assessment and supplementation based on deficiencies. "
            "Focus on a balanced diet rich in iron, B12, and folate."
        )
    
    lifestyle_changes = [
        "Maintain regular exercise within your comfort level",
        "Ensure adequate sleep and stress management",
        "Stay well-hydrated",
        "Avoid smoking and limit alcohol consumption"
    ]
    recommendation_parts.append("Lifestyle Recommendations: " + " ".join(lifestyle_changes))
    
    monitoring = "Schedule regular follow-up blood tests to monitor your progress."
    recommendation_parts.append(monitoring)
    
    return " ".join(recommendation_parts)

# -------------------- Hepatitis C Models --------------------
class HepCInput(BaseModel):
    sex: int = Field(..., description="Sex (0 for Female, 1 for Male)")
    age: int = Field(..., description="Age in years")
    alb: float = Field(..., description="Albumin (g/L)")
    alp: float = Field(..., description="Alkaline Phosphatase (U/L)")
    alt: float = Field(..., description="Alanine Aminotransferase (U/L)")
    ast: float = Field(..., description="Aspartate Aminotransferase (U/L)")
    bil: float = Field(..., description="Bilirubin (mg/dL)")
    che: float = Field(..., description="Cholinesterase (U/L)")
    chol: float = Field(..., description="Cholesterol (mg/dL)")
    crea: float = Field(..., description="Creatinine (mg/dL)")
    ggt: float = Field(..., description="Gamma-Glutamyl Transferase (U/L)")
    prot: float = Field(..., description="Total Protein (g/L)")

    @model_validator(mode='after')
    def validate_fields(self) -> 'HepCInput':
        if self.sex not in [0, 1]:
            raise ValueError('Sex must be 0 (Female) or 1 (Male)')
        if self.age < 0 or self.age > 120:
            raise ValueError('Age must be between 0 and 120 years')

        for field_name, value in {
            'alb': self.alb,
            'alp': self.alp,
            'alt': self.alt,
            'ast': self.ast,
            'bil': self.bil,
            'che': self.che,
            'chol': self.chol,
            'crea': self.crea,
            'ggt': self.ggt,
            'prot': self.prot
        }.items():
            if value <= 0:
                raise ValueError(f'{field_name} must be positive')

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "sex": 0,
                "age": 45,
                "alb": 38,
                "alp": 70,
                "alt": 30,
                "ast": 25,
                "bil": 0.8,
                "che": 8500,
                "chol": 180,
                "crea": 0.9,
                "ggt": 35,
                "prot": 70
            }
        }
    }

class HepCPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    disease_stage: Dict[str, str]
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]
 
class HepCParameterRanges:
    def __init__(self):
        self.ranges = {
            'alb': {'low': 35, 'high': 50, 'unit': 'g/L'},
            'alp': {'low': 30, 'high': 120, 'unit': 'U/L'},
            'alt': {'low': 7, 'high': 56, 'unit': 'U/L'},
            'ast': {'low': 8, 'high': 48, 'unit': 'U/L'},
            'bil': {'low': 3, 'high': 21, 'unit': 'µmol/L'},
            'che': {'low': 5, 'high': 12, 'unit': 'kU/L'},
            'chol': {'low': 3.6, 'high': 5.2, 'unit': 'mmol/L'},
            'crea': {'low': 62, 'high': 106, 'unit': 'µmol/L'},
            'ggt': {'low': 9, 'high': 50, 'unit': 'U/L'},
            'prot': {'low': 64, 'high': 83, 'unit': 'g/L'}
        }
    
    def get_hepc_range(self, parameter: str) -> dict:
        if parameter not in self.ranges:
            raise ValueError(f"Unknown parameter: {parameter}")
        return self.ranges[parameter]

        
def get_disease_stage(prediction: int) -> Dict[str, str]:    
    stages = {        
        0: {                
            "stage": "Blood Donor",                
            "description": "Healthy individual with normal liver function. No evidence of hepatitis C infection.",
            "medical_implications": "Regular health monitoring recommended. Continue routine blood donor screening.",
            "typical_management": "Maintain healthy lifestyle and regular check-ups."   
           },                
        1: {            
            "stage": "Suspected Blood Donor",            
            "description": "Individual with some abnormal liver function tests requiring further investigation.",                               
            "medical_implications": "May indicate early liver changes or other health conditions. Further testing needed.",            
            "typical_management": "Additional screening tests and closer monitoring of liver function."        
            },        
        2: {            
            "stage": "Hepatitis",            
            "description": "Active hepatitis C infection with liver inflammation.",            
            "medical_implications": "Active viral infection affecting liver function. Risk of liver damage.",            
            "typical_management": "Antiviral therapy evaluation, regular monitoring of liver function, lifestyle modifications."        
            },        
        3: {            
            "stage": "Fibrosis",            
            "description": "Development of liver scarring due to chronic inflammation.",            
            "medical_implications": "Progressive liver damage with reduced liver function. Risk of further progression.",            
            "typical_management": "Antiviral therapy if not already treated, regular liver imaging, lifestyle modifications, specialist monitoring."
            },
        4: {            
            "stage": "Liver Cirrhosis",            
            "description": "Advanced liver scarring with significant impact on liver function.",            
            "medical_implications": "Severe liver damage with complications risk. Regular screening for liver cancer needed.",
            "typical_management": "Intensive monitoring, complications management, possible transplant evaluation."
            }
    }
    return stages.get(prediction, {        
        "stage": "Unknown",        
        "description": "Classification not recognized",        
        "medical_implications": "Unable to determine",
        "typical_management": "Please consult healthcare provider"
    })

def analyze_hepc_parameter(param_name: str, value: float, ranges: HepCParameterRanges) -> DetailedAnalysis:
    param_range = ranges.get_hepc_range(param_name)
    status = "Normal"

    parameter_advice = {
        'alb': {
            'low': {
                'interpretation': "Low albumin indicates possible liver dysfunction, malnutrition, or inflammation.",
                'recommendations': [
                    "Increase protein intake through lean meats, eggs, and legumes",
                    "Consult with a nutritionist for dietary planning",
                    "Monitor for fluid retention or edema",
                    "Schedule follow-up liver function tests",
                    "Consider vitamin D supplementation as it may help albumin synthesis"
                ]
            },
            'high': {
                'interpretation': "High albumin might indicate dehydration.",
                'recommendations': [
                    "Increase fluid intake",
                    "Monitor hydration status",
                    "Consider underlying causes of dehydration",
                    "Review medication side effects that may affect hydration"
                ]
            },
            'normal': {
                'interpretation': "Albumin levels are within normal range, indicating good liver synthetic function.",
                'recommendations': [
                    "Maintain current dietary protein intake",
                    "Continue regular exercise routine",
                    "Schedule routine follow-up as recommended"
                ]
            }
        },
        'alp': {
            'low': {
                'interpretation': "Low ALP might indicate malnutrition, mineral deficiencies, or rare genetic conditions.",
                'recommendations': [
                    "Evaluate zinc and magnesium levels",
                    "Assess vitamin D status",
                    "Consider bone density testing",
                    "Review current medications that might affect ALP"
                ]
            },
            'high': {
                'interpretation': "Elevated ALP may indicate liver disease, bone disorders, or biliary obstruction.",
                'recommendations': [
                    "Schedule liver imaging studies",
                    "Consider bone disease evaluation",
                    "Assess vitamin D levels",
                    "Monitor other liver enzymes closely",
                    "Avoid alcohol consumption"
                ]
            },
            'normal': {
                'interpretation': "ALP levels are within normal range, suggesting healthy liver and bone function.",
                'recommendations': [
                    "Maintain regular exercise for bone health",
                    "Ensure adequate calcium and vitamin D intake",
                    "Continue routine monitoring"
                ]
            }
        },
        'alt': {
            'low': {
                'interpretation': "Low ALT is usually not clinically significant but might indicate vitamin B6 deficiency.",
                'recommendations': [
                    "Review B-vitamin intake",
                    "Maintain balanced diet",
                    "Continue routine monitoring",
                    "Discuss with healthcare provider if persistent"
                ]
            },
            'high': {
                'interpretation': "Elevated ALT indicates possible liver cell damage or inflammation.",
                'recommendations': [
                    "Immediate consultation with hepatologist",
                    "Stop alcohol consumption completely",
                    "Avoid hepatotoxic medications",
                    "Schedule comprehensive liver function tests",
                    "Consider viral hepatitis testing",
                    "Request liver ultrasound"
                ]
            },
            'normal': {
                'interpretation': "ALT is within normal range, suggesting minimal liver cell damage.",
                'recommendations': [
                    "Maintain healthy liver through diet and exercise",
                    "Limit alcohol intake",
                    "Continue regular monitoring"
                ]
            }
        },
        'ast': {
            'low': {
                'interpretation': "Low AST is generally not clinically significant.",
                'recommendations': [
                    "Maintain current healthy habits",
                    "Continue routine monitoring",
                    "No specific intervention needed"
                ]
            },
            'high': {
                'interpretation': "Elevated AST suggests liver cell damage, muscle damage, or other organ involvement.",
                'recommendations': [
                    "Urgent hepatology consultation",
                    "Complete alcohol cessation",
                    "Review all medications with doctor",
                    "Schedule comprehensive metabolic panel",
                    "Consider cardiac enzyme testing",
                    "Request abdominal imaging"
                ]
            },
            'normal': {
                'interpretation': "AST within normal range indicates minimal tissue damage.",
                'recommendations': [
                    "Continue liver-healthy lifestyle",
                    "Maintain regular exercise routine",
                    "Schedule routine follow-up"
                ]
            }
        },
        'bil': {
            'low': {
                'interpretation': "Low bilirubin is generally not clinically significant.",
                'recommendations': [
                    "No specific intervention needed",
                    "Continue current healthy habits",
                    "Maintain routine monitoring"
                ]
            },
            'high': {
                'interpretation': "Elevated bilirubin may indicate liver dysfunction, bile duct obstruction, or hemolysis.",
                'recommendations': [
                    "Urgent medical evaluation needed",
                    "Schedule liver and biliary tract imaging",
                    "Complete blood count to check for hemolysis",
                    "Avoid hepatotoxic medications",
                    "Monitor for jaundice and dark urine",
                    "Consider genetic testing for Gilbert's syndrome"
                ]
            },
            'normal': {
                'interpretation': "Bilirubin within normal range suggests good liver function and normal red blood cell turnover.",
                'recommendations': [
                    "Maintain healthy diet and exercise",
                    "Continue routine monitoring",
                    "Stay well-hydrated"
                ]
            }
        },
        'che': {
            'low': {
                'interpretation': "Low cholinesterase might indicate decreased liver synthetic function or exposure to toxins.",
                'recommendations': [
                    "Evaluate for liver disease",
                    "Review possible toxin exposures",
                    "Check protein synthesis markers",
                    "Consider pesticide exposure assessment",
                    "Schedule follow-up liver function tests"
                ]
            },
            'high': {
                'interpretation': "Elevated cholinesterase might be seen in early stages of liver disease or diabetes.",
                'recommendations': [
                    "Check blood glucose levels",
                    "Evaluate thyroid function",
                    "Monitor other liver enzymes",
                    "Consider diabetes screening",
                    "Review diet and lifestyle factors"
                ]
            },
            'normal': {
                'interpretation': "Normal cholinesterase indicates good liver synthetic function.",
                'recommendations': [
                    "Continue current healthy habits",
                    "Maintain regular monitoring",
                    "Follow routine screening schedule"
                ]
            }
        },
        'chol': {
            'low': {
                'interpretation': "Low cholesterol might indicate malnutrition or advanced liver disease.",
                'recommendations': [
                    "Evaluate nutritional status",
                    "Check other liver function tests",
                    "Consider malabsorption evaluation",
                    "Review medications affecting cholesterol",
                    "Monitor diet and weight"
                ]
            },
            'high': {
                'interpretation': "Elevated cholesterol increases cardiovascular risk and may be linked to liver disease.",
                'recommendations': [
                    "Implement heart-healthy diet",
                    "Increase physical activity",
                    "Consider statin therapy consultation",
                    "Monitor blood pressure",
                    "Schedule regular cardiovascular checkups",
                    "Assess family history of heart disease"
                ]
            },
            'normal': {
                'interpretation': "Cholesterol within normal range suggests good metabolic health.",
                'recommendations': [
                    "Maintain healthy diet and exercise",
                    "Continue heart-healthy lifestyle",
                    "Schedule routine lipid monitoring"
                ]
            }
        },
        'crea': {
            'low': {
                'interpretation': "Low creatinine might indicate decreased muscle mass or liver disease.",
                'recommendations': [
                    "Evaluate muscle mass and nutrition",
                    "Consider protein intake assessment",
                    "Review exercise routine",
                    "Monitor kidney function",
                    "Check for muscle wasting"
                ]
            },
            'high': {
                'interpretation': "Elevated creatinine suggests possible kidney dysfunction.",
                'recommendations': [
                    "Urgent nephrology consultation",
                    "Monitor fluid intake and output",
                    "Check blood pressure regularly",
                    "Review medications affecting kidneys",
                    "Consider renal ultrasound",
                    "Assess for dehydration"
                ]
            },
            'normal': {
                'interpretation': "Normal creatinine indicates good kidney function.",
                'recommendations': [
                    "Maintain good hydration",
                    "Continue regular exercise",
                    "Monitor blood pressure",
                    "Schedule routine kidney function tests"
                ]
            }
        },
        'ggt': {
            'low': {
                'interpretation': "Low GGT is generally not clinically significant.",
                'recommendations': [
                    "No specific intervention needed",
                    "Continue current healthy habits",
                    "Maintain routine monitoring"
                ]
            },
            'high': {
                'interpretation': "Elevated GGT suggests liver disease, alcohol use, or medication effects.",
                'recommendations': [
                    "Stop alcohol consumption",
                    "Review all medications",
                    "Schedule liver imaging",
                    "Monitor other liver enzymes",
                    "Consider alcohol counseling if relevant",
                    "Evaluate for fatty liver disease"
                ]
            },
            'normal': {
                'interpretation': "Normal GGT suggests healthy liver function.",
                'recommendations': [
                    "Maintain alcohol-free or moderate drinking",
                    "Continue healthy diet",
                    "Schedule routine monitoring"
                ]
            }
        },
        'prot': {
            'low': {
                'interpretation': "Low total protein might indicate malnutrition, liver disease, or protein loss.",
                'recommendations': [
                    "Increase dietary protein intake",
                    "Evaluate for protein loss conditions",
                    "Consider nutritionist consultation",
                    "Monitor albumin levels",
                    "Check for malabsorption"
                ]
            },
            'high': {
                'interpretation': "High total protein might indicate infection, inflammation, or certain blood disorders.",
                'recommendations': [
                    "Evaluate for chronic inflammation",
                    "Check immunoglobulin levels",
                    "Consider bone marrow evaluation",
                    "Monitor for dehydration",
                    "Schedule follow-up testing"
                ]
            },
            'normal': {
                'interpretation': "Normal total protein indicates good nutritional and liver synthetic status.",
                'recommendations': [
                    "Maintain balanced diet",
                    "Continue regular exercise",
                    "Schedule routine monitoring"
                ]
            }
        }
    }

    if value < param_range['low']:
        status = "Low"
        interpretation = parameter_advice[param_name]['low']['interpretation']
        recommendations = parameter_advice[param_name]['low']['recommendations']
    elif value > param_range['high']:
        status = "High"
        interpretation = parameter_advice[param_name]['high']['interpretation']
        recommendations = parameter_advice[param_name]['high']['recommendations']
    else:
        interpretation = parameter_advice[param_name]['normal']['interpretation']
        recommendations = parameter_advice[param_name]['normal']['recommendations']

    return DetailedAnalysis(
        parameter=param_name.upper(),
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_hepc_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_hepc_recommendation(detailed_analyses: List[DetailedAnalysis], risk_level: str) -> str:
    abnormal_params = [analysis for analysis in detailed_analyses if analysis.status != "Normal"]

    if not abnormal_params:
        return ("Your liver function tests are within normal ranges. Continue maintaining a healthy lifestyle "
                "and schedule regular check-ups for monitoring.")

    recommendation_parts = []

    if risk_level == "High Risk":
        recommendation_parts.append("URGENT: Immediate consultation with a hepatologist is recommended.")

    if any(analysis.parameter in ['ALT', 'AST'] and analysis.status == "High" for analysis in abnormal_params):
        recommendation_parts.append(
            "Priority: Elevated liver enzymes indicate possible liver damage. "
            "Avoid alcohol and hepatotoxic medications. Schedule comprehensive liver function tests."
        )

    if any(analysis.parameter == 'bil' and analysis.status == "High" for analysis in abnormal_params):
        recommendation_parts.append(
            "Important: Elevated bilirubin requires prompt medical attention. "
            "Consider additional imaging studies."
        )

    lifestyle_changes = [
        "Maintain a healthy diet low in processed foods",
        "Avoid alcohol consumption",
        "Exercise regularly within your limits",
        "Get adequate rest and manage stress",
        "Avoid sharing personal items that may transmit blood"
    ]
    recommendation_parts.append("Lifestyle Recommendations: " + " ".join(lifestyle_changes))

    monitoring = "Schedule regular follow-up appointments and liver function tests as recommended by your healthcare provider."
    recommendation_parts.append(monitoring)

    return " ".join(recommendation_parts)


# -------------------- Kidney Models --------------------

class KidneyInput(BaseModel):
    specific_gravity: float = Field(..., description="Specific Gravity of urine")
    albumin: float = Field(..., description="Albumin level (0-5 scale)")
    red_blood_cells: int = Field(..., description="Red Blood Cells in urine (0=normal, 1=abnormal)")
    pus_cell: int = Field(..., description="Pus Cell in urine (0=normal, 1=abnormal)")
    blood_urea: float = Field(..., description="Blood Urea (mg/dL)")
    serum_creatinine: float = Field(..., description="Serum Creatinine (mg/dL)")
    sodium: float = Field(..., description="Sodium (mEq/L)")
    potassium: float = Field(..., description="Potassium (mEq/L)")
    hemoglobin: float = Field(..., description="Hemoglobin (g/dL)")
    packed_cell_volume: float = Field(..., description="Packed Cell Volume (%)")
    hypertension: int = Field(..., description="Presence of Hypertension (0=no, 1=yes)")
    diabetes_mellitus: int = Field(..., description="Presence of Diabetes Mellitus (0=no, 1=yes)")
    anemia: int = Field(..., description="Presence of Anemia (0=no, 1=yes)")
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'KidneyInput':
        # Validate numeric fields
        if not (1.000 <= self.specific_gravity <= 1.050):
            raise ValueError('Specific gravity should typically be between 1.000 and 1.050')
        
        if not (0 <= self.albumin <= 5):
            raise ValueError('Albumin must be between 0 and 5')
        
        # Validate binary fields
        binary_fields = ['red_blood_cells', 'pus_cell', 'hypertension', 'diabetes_mellitus', 'anemia']
        for field in binary_fields:
            value = getattr(self, field)
            if value not in [0, 1]:
                raise ValueError(f'{field} must be either 0 or 1')
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "specific_gravity": 1.015,
                "albumin": 2.0,
                "red_blood_cells": 1,
                "pus_cell": 1,
                "blood_urea": 100.0,
                "serum_creatinine": 4.0,
                "sodium": 135.0,
                "potassium": 5.2,
                "hemoglobin": 10.0,
                "packed_cell_volume": 30.0,
                "hypertension": 1,
                "diabetes_mellitus": 1,
                "anemia": 1
            }
        }
    }
    
class KidneyPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]

class KidneyParameterRanges:
    """Defines normal ranges for kidney-related parameters"""
    
    def __init__(self):
        self.ranges = {
            'specific_gravity': {'low': 1.010, 'high': 1.025},
            'albumin': {'low': 0, 'high': 0},  # Ideally 0 in urine
            'blood_urea': {'low': 7, 'high': 20},
            'serum_creatinine': {'low': 0.6, 'high': 1.2},
            'sodium': {'low': 135, 'high': 145},
            'potassium': {'low': 3.5, 'high': 5.0},
            'hemoglobin': {'low': 13.5, 'high': 17.5},  # For males, different for females
            'packed_cell_volume': {'low': 38, 'high': 50},  # For males, different for females
        }

        # Binary parameters expected values (what's considered "normal")
        self.binary_params = {
            'red_blood_cells': 0,  # Changed to 0 for normal
            'pus_cell': 0,         # Changed to 0 for normal
            'hypertension': 0,     # Changed to 0 for no
            'diabetes_mellitus': 0, # Changed to 0 for no
            'anemia': 0            # Changed to 0 for no
        }
        

    def get_kidney_range(self, parameter: str) -> Dict[str, float]:
        return self.ranges.get(parameter, {'low': 0, 'high': 0})
    
    def get_binary_normal(self, parameter: str) -> int:
        return self.binary_params.get(parameter, 0)
    
def analyze_kidney_numeric_parameter(param_name: str, value: float, ranges: KidneyParameterRanges) -> DetailedAnalysis:
    """Analyze a numeric parameter and provide recommendations"""
    
    param_range = ranges.get_kidney_range(param_name)
    status = "Normal"
    interpretation = ""
    recommendations = []
    
    # Parameter-specific analysis
    if param_name == 'specific_gravity':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low urine specific gravity may indicate diluted urine or kidney's inability to concentrate urine."
            recommendations = [
                "Evaluate hydration status",
                "Further testing for diabetes insipidus",
                "Monitor fluid intake"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "High specific gravity indicates concentrated urine, possibly due to dehydration or kidney stress."
            recommendations = [
                "Increase fluid intake",
                "Monitor for signs of dehydration",
                "Follow up with healthcare provider"
            ]
        else:
            interpretation = "Urine specific gravity is within normal range."
            recommendations = ["Maintain adequate hydration"]

    elif param_name == 'albumin':
        if value > param_range['high']:
            status = "High"
            interpretation = "Presence of albumin in urine (albuminuria) indicates kidney damage."
            recommendations = [
                "Blood pressure and blood sugar control",
                "Dietary protein restriction may be advised",
                "Regular kidney function monitoring",
                "Consult nephrologist"
            ]
        else:
            interpretation = "No albumin detected in urine, which is normal."
            recommendations = ["Continue kidney-friendly lifestyle"]

    elif param_name == 'blood_urea':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low blood urea may indicate liver problems or malnutrition."
            recommendations = [
                "Protein intake assessment",
                "Liver function evaluation",
                "Nutritional consultation"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "Elevated blood urea indicates decreased kidney function or dehydration."
            recommendations = [
                "Increase fluid intake",
                "Reduce protein intake",
                "Medication review with healthcare provider",
                "Regular kidney function monitoring"
            ]
        else:
            interpretation = "Blood urea is within normal range."
            recommendations = ["Continue kidney-friendly diet and hydration"]
                
    elif param_name == 'serum_creatinine':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low serum creatinine may indicate decreased muscle mass."
            recommendations = [
                "Protein intake evaluation",
                "Consider strength training if appropriate",
                "Nutritional assessment"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "Elevated serum creatinine indicates impaired kidney function."
            recommendations = [
                "Kidney function monitoring",
                "Blood pressure control",
                "Dietary modifications (reduce protein, sodium)",
                "Consult with nephrologist"
            ]
        else:
            interpretation = "Serum creatinine is within normal range."
            recommendations = ["Regular kidney health monitoring"]
                
    elif param_name == 'sodium':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low sodium (hyponatremia) may affect nerve and muscle function."
            recommendations = [
                "Balanced electrolyte intake",
                "Moderate salt consumption",
                "Proper hydration",
                "Medication review with healthcare provider"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "High sodium (hypernatremia) indicates dehydration or excessive sodium intake."
            recommendations = [
                "Increase water intake",
                "Reduce dietary sodium",
                "Monitor for signs of dehydration",
                "Follow up with healthcare provider"
            ]
        else:
            interpretation = "Sodium level is within normal range."
            recommendations = ["Continue balanced sodium intake"]
                
    elif param_name == 'potassium':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low potassium (hypokalemia) affects heart and muscle function."
            recommendations = [
                "Increase potassium-rich foods (bananas, potatoes, etc.)",
                "Medication review",
                "Electrolyte monitoring",
                "Consult healthcare provider"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "High potassium (hyperkalemia) is dangerous for heart rhythm."
            recommendations = [
                "Reduce potassium-rich foods",
                "Medication review is critical",
                "Emergency medical attention if symptomatic",
                "Regular monitoring with healthcare provider"
            ]
        else:
            interpretation = "Potassium level is within normal range."
            recommendations = ["Continue balanced diet"]
                
    elif param_name == 'hemoglobin':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low hemoglobin indicates anemia, common in kidney disease."
            recommendations = [
                "Iron-rich diet",
                "Consider iron supplementation",
                "Erythropoietin evaluation",
                "Regular blood count monitoring"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "Elevated hemoglobin may indicate polycythemia or dehydration."
            recommendations = [
                "Increase hydration",
                "Evaluate for polycythemia",
                "Regular monitoring"
            ]
        else:
            interpretation = "Hemoglobin level is within normal range."
            recommendations = ["Continue balanced diet with iron-rich foods"]
                
    elif param_name == 'packed_cell_volume':
        if value < param_range['low']:
            status = "Low"
            interpretation = "Low PCV indicates anemia, common in kidney disease."
            recommendations = [
                "Iron-rich diet",
                "Consider iron supplementation",
                "Erythropoietin evaluation",
                "Regular blood count monitoring"
            ]
        elif value > param_range['high']:
            status = "High"
            interpretation = "Elevated PCV may indicate dehydration or polycythemia."
            recommendations = [
                "Increase hydration",
                "Evaluate for polycythemia",
                "Regular monitoring"
            ]
        else:
            interpretation = "PCV is within normal range."
            recommendations = ["Continue balanced diet with iron-rich foods"]
                
    else:
        interpretation = f"{param_name} value recorded."
        recommendations = ["Discuss with healthcare provider"]

    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def analyze_kidney_categorical_parameter(param_name: str, value: str, ranges: ParameterRanges) -> DetailedAnalysis:
    """Analyze a categorical parameter and provide recommendations"""
    
    normal_value = ranges.get_binary_normal(param_name)
    status = "Normal" if value == normal_value else "Abnormal"
    interpretation = ""
    recommendations = []

    
    # Parameter-specific analysis
    if param_name == 'red_blood_cells':
        if value != normal_value:
            interpretation = "Abnormal RBCs in urine indicate kidney damage or urinary tract issues."
            recommendations = [
                "Complete urinalysis evaluation",
                "Kidney function assessment",
                "Urinary tract evaluation",
                "Consult with nephrologist"
            ]
        else:
            interpretation = "Normal RBCs in urine."
            recommendations = ["Continue routine kidney health monitoring"]
            
    elif param_name == 'pus_cell':
        if value != normal_value:
            interpretation = "Presence of pus cells in urine indicates infection or inflammation."
            recommendations = [
                "Urine culture may be needed",
                "Antibiotic evaluation",
                "Increased fluid intake",
                "Urinary tract evaluation"
            ]
        else:
            interpretation = "Normal pus cell count in urine."
            recommendations = ["Maintain good hydration"]
                

    elif param_name == 'hypertension':
        if value != normal_value:
            interpretation = "Hypertension is a major risk factor for kidney disease progression."
            recommendations = [
                "Regular blood pressure monitoring",
                "Low sodium DASH diet",
                "Regular physical activity",
                "Medication adherence"
            ]
        else:
            interpretation = "No hypertension reported."
            recommendations = ["Continue heart-healthy lifestyle"]
            
    elif param_name == 'diabetes_mellitus':
        if value != normal_value:
            interpretation = "Diabetes is a major risk factor for kidney disease."
            recommendations = [
                "Tight glucose control",
                "Regular kidney function monitoring",
                "ACE inhibitor or ARB medication evaluation",
                "Diet and exercise management"
            ]
        else:
            interpretation = "No diabetes reported."
            recommendations = ["Continue healthy lifestyle to prevent diabetes"]

    elif param_name == 'anemia':
        if value != normal_value:
            interpretation = "Anemia is common in kidney disease due to reduced erythropoietin."
            recommendations = [
                "Iron supplementation evaluation",
                "Erythropoietin therapy consideration",
                "Nutritional support",
                "Regular blood count monitoring"
            ]
        else:
            interpretation = "No anemia reported."
            recommendations = ["Continue iron-rich diet"]
    else:
        interpretation = f"{param_name} status recorded."
        recommendations = ["Discuss with healthcare provider"]

    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_kidney_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"
    
def generate_kidney_overall_recommendation(detailed_analyses: List[DetailedAnalysis], risk_level: str) -> str:
    # Find abnormal parameters
    abnormal_params = [analysis for analysis in detailed_analyses if analysis.status != "Normal"]
    
    if not abnormal_params:
        return ("Your kidney health parameters are within normal ranges. Continue maintaining a healthy diet, "
                "proper hydration, and lifestyle. Regular check-ups are recommended for monitoring.")
    
    recommendation_parts = []
        
    if risk_level == "High Risk":
        recommendation_parts.append("URGENT: Consult with a nephrologist (kidney specialist) for comprehensive evaluation.")
    elif risk_level == "Moderate Risk":
        recommendation_parts.append("IMPORTANT: Schedule an appointment with your healthcare provider for kidney function assessment.")
    else:
        recommendation_parts.append("Follow up with your healthcare provider to discuss these results.")
    
    # Critical parameter issues
    critical_issues = any(analysis.parameter in ['serum_creatinine', 'blood_urea', 'albumin'] and analysis.status != "Normal" 
                        for analysis in abnormal_params)
    if critical_issues:
        recommendation_parts.append(
            "Priority: Address abnormal kidney function indicators through medical management. "
            "Follow medication regimens precisely and attend all follow-up appointments."
        )
    
    # Lifestyle recommendations based on common issues
    lifestyle_recs = []
    
    
    # Fluid management
    if any(analysis.parameter in ['specific_gravity', 'blood_urea', 'sodium'] and analysis.status != "Normal" 
          for analysis in abnormal_params):
        lifestyle_recs.append("Maintain proper hydration (typically 2-3 liters daily unless restricted)")
    
    # Diet recommendations
    diet_issues = any(analysis.parameter in ['albumin', 'blood_urea', 'serum_creatinine', 'potassium', 'sodium'] 
                     and analysis.status != "Normal" for analysis in abnormal_params)
    if diet_issues:
        lifestyle_recs.append("Follow a kidney-friendly diet (typically lower in protein, sodium, and potassium)")
    
    # Blood pressure management
    if any(analysis.parameter in ['hypertension'] and analysis.status != "Normal" 
          for analysis in abnormal_params):
        lifestyle_recs.append("Monitor blood pressure regularly and maintain blood pressure control")
    
    # Blood sugar management
    if any(analysis.parameter in ['diabetes_mellitus'] and analysis.status != "Normal" 
          for analysis in abnormal_params):
        lifestyle_recs.append("Maintain tight blood sugar control and regular diabetes management")
    
    # Add general lifestyle recommendations
    general_recs = [
        "Maintain regular physical activity appropriate for your condition",
        "Avoid nephrotoxic medications (NSAIDs like ibuprofen, certain antibiotics)",
        "Stop smoking and limit alcohol consumption",
        "Manage stress through appropriate techniques"
    ]
        
    # Combine all lifestyle recommendations
    if lifestyle_recs:
        recommendation_parts.append("Key Focus Areas: " + " ".join(lifestyle_recs))
    recommendation_parts.append("General Recommendations: " + " ".join(general_recs))
    
    # Monitoring recommendation
    monitoring = "Schedule regular kidney function tests and urinalysis as recommended by your healthcare provider."
    recommendation_parts.append(monitoring)
    
    return " ".join(recommendation_parts)

def prepare_kidney_features(data: KidneyInput) -> np.ndarray:

    # Create feature array
    features = np.array([[
        data.specific_gravity,
        data.albumin,
        data.red_blood_cells,
        data.pus_cell,
        data.blood_urea,
        data.serum_creatinine,
        data.sodium,
        data.potassium,
        data.hemoglobin,
        data.packed_cell_volume,
        data.hypertension,
        data.diabetes_mellitus,
        data.anemia
    ]])
    
    return features
# -------------------- Heart canser Models --------------------
class HeartDiseaseInput(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (0 for Female, 1 for Male)")
    cp: int = Field(..., description="Chest Pain Type (0-3)")
    trestbps: int = Field(..., description="Resting Blood Pressure (mm Hg)")
    chol: int = Field(..., description="Serum Cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., description="Resting ECG results (0-2)")
    thalach: int = Field(..., description="Maximum Heart Rate Achieved")
    exang: int = Field(..., description="Exercise Induced Angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., description="ST Depression Induced by Exercise Relative to Rest")
    slope: int = Field(..., description="Slope of the Peak Exercise ST Segment (0-2)")
    ca: int = Field(..., description="Number of Major Vessels Colored by Fluoroscopy (0-3)")
    thal: int = Field(..., description="Thalassemia (0-3)")
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'HeartDiseaseInput':
        if self.sex not in [0, 1]:
            raise ValueError('Sex must be 0 (Female) or 1 (Male)')
        
        if self.cp not in [0, 1, 2, 3]:
            raise ValueError('Chest Pain Type must be between 0 and 3')
        
        if self.fbs not in [0, 1]:
            raise ValueError('Fasting Blood Sugar must be 0 or 1')
        
        if self.restecg not in [0, 1, 2]:
            raise ValueError('Resting ECG results must be between 0 and 2')
        
        if self.exang not in [0, 1]:
            raise ValueError('Exercise Induced Angina must be 0 or 1')
        
        if self.slope not in [0, 1, 2]:
            raise ValueError('Slope must be between 0 and 2')
        
        if not 0 <= self.ca <= 3:
            raise ValueError('Number of Major Vessels must be between 0 and 3')
        
        if not 0 <= self.thal <= 3:
            raise ValueError('Thalassemia must be between 0 and 3')
        
        if self.age <= 0 or self.age > 120:
            raise ValueError('Age must be positive and realistic')
                
        if self.trestbps <= 0:
            raise ValueError('Resting Blood Pressure must be positive')
        
        if self.chol <= 0:
            raise ValueError('Cholesterol must be positive')
        
        if self.thalach <= 0:
            raise ValueError('Maximum Heart Rate must be positive')
        
        return self
        
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 53,
                "sex": 1,
                "cp": 0,
                "trestbps": 140,
                "chol": 203,
                "fbs": 1,
                "restecg": 0,
                "thalach": 155,
                "exang": 1,
                "oldpeak": 3.1,
                "slope": 0,
                "ca": 0,
                "thal": 3
            }
        }
    }
    
class HeartDiseasePrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]

class HeartParameterRanges:
    def __init__(self):
        self.ranges = {
            'age': {'low': 18, 'high': 65},
            'trestbps': {'low': 90, 'high': 120},
            'chol': {'low': 130, 'high': 200},
            'thalach': {'low': 60, 'high': 220, 'optimal': lambda age: 220 - age},
            'oldpeak': {'low': 0, 'high': 0.5}
        }
    def get_heart_range(self, parameter: str, age: int = None) -> Dict[str, float]:
        result = self.ranges.get(parameter, {'low': 0, 'high': 0})
        
        # For max heart rate, we calculate optimal based on age
        if parameter == 'thalach' and age is not None:
            result['optimal'] = 220 - age
            
        return result

def analyze_heart_parameter(param_name: str, value: Any, ranges: HeartParameterRanges, age: int = None) -> DetailedAnalysis:
    status = "Normal"
    interpretation = ""
    recommendations = []
    
    # Numeric parameters
    if param_name in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        param_range = ranges.get_heart_range(param_name, age)
        
        if param_name == 'age':
            status = "Risk Factor" if value > param_range['high'] else "Normal"
            interpretation = "Age is a non-modifiable risk factor for heart disease."
            recommendations = ["Regular cardiovascular check-ups", "Focus on modifiable risk factors like diet and exercise"]
            
        elif param_name == 'trestbps':
            if value < param_range['low']:
                status = "Low"
                interpretation = "Low blood pressure may cause dizziness but is generally not a heart disease risk."
                recommendations = ["Monitor for symptoms like dizziness", "Stay hydrated", "Consult with healthcare provider"]
            elif value > param_range['high']:
                status = "High"
                interpretation = "Elevated blood pressure increases risk of heart disease and stroke."
                recommendations = [
                    "Reduce sodium intake",
                    "Maintain healthy weight",
                    "Regular exercise",
                    "Consider medication if consistently high",
                    "Monitor blood pressure regularly"
                ]
            else:
                interpretation = "Blood pressure is within normal range."
                recommendations = ["Continue current healthy lifestyle", "Monitor periodically"]
                
        elif param_name == 'chol':
            if value < param_range['low']:
                status = "Low"
                interpretation = "Low cholesterol is generally not a concern for heart disease."
                recommendations = ["Maintain current diet", "Regular check-ups"]
            elif value > param_range['high']:
                status = "High"
                interpretation = "Elevated cholesterol increases risk of atherosclerosis and heart disease."
                recommendations = [
                    "Reduce saturated fat intake",
                    "Increase fiber in diet",
                    "Regular exercise",
                    "Consider medication if consistently high",
                    "Follow-up cholesterol panel recommended"
                ]
            else:
                interpretation = "Cholesterol is within normal range."
                recommendations = ["Continue heart-healthy diet", "Regular monitoring"]
                
        elif param_name == 'thalach':
            optimal = param_range['optimal']
            if value < optimal * 0.7:  # Less than 70% of max
                status = "Low"
                interpretation = "Lower maximum heart rate might indicate reduced cardiac fitness."
                recommendations = [
                    "Gradual cardiovascular training",
                    "Consult cardiologist before starting exercise program",
                    "Consider stress test"
                ]
            elif value > optimal:
                status = "High"
                interpretation = "Higher than expected maximum heart rate may indicate stress or other factors."
                recommendations = ["Cardiovascular evaluation recommended", "Consider stress management techniques"]
            else:
                interpretation = "Maximum heart rate is within expected range."
                recommendations = ["Continue regular exercise", "Maintain cardiac fitness"]
                
        elif param_name == 'oldpeak':
            if value > param_range['high']:
                status = "Elevated"
                interpretation = "ST depression suggests possible myocardial ischemia."
                recommendations = [
                    "Immediate cardiovascular evaluation",
                    "Discuss results with cardiologist",
                    "Further cardiac testing indicated"
                ]
            else:
                interpretation = "ST depression is within normal limits."
                recommendations = ["Continue regular cardiac monitoring"]
    
    # Categorical parameters
    elif param_name == 'sex':
        status = "Risk Factor" if value == 1 else "Normal"
        interpretation = "Biological males have statistically higher risk of heart disease."
        recommendations = ["Regular cardiovascular check-ups", "Be vigilant about other risk factors"]
    
    elif param_name == 'cp':
        if value == 0:
            status = "Typical Angina"
            interpretation = "Typical angina is strongly associated with coronary artery disease."
            recommendations = ["Urgent cardiac evaluation", "Stress test or angiography may be indicated"]
        elif value == 1:
            status = "Atypical Angina"
            interpretation = "Atypical angina may indicate cardiac or non-cardiac issues."
            recommendations = ["Cardiac evaluation recommended", "Consider additional testing"]
        elif value == 2:
            status = "Non-anginal Pain"
            interpretation = "Non-anginal pain is less likely to be cardiac in origin."
            recommendations = ["Monitor symptoms", "Evaluate for non-cardiac causes"]
        elif value == 3:
            status = "Asymptomatic"
            interpretation = "No chest pain reported."
            recommendations = ["Continue preventive measures", "Report any new symptoms promptly"]
    
    elif param_name == 'fbs':
        if value == 1:
            status = "Elevated"
            interpretation = "Elevated fasting blood sugar indicates possible diabetes, a cardiac risk factor."
            recommendations = [
                "Blood glucose monitoring",
                "Dietary modifications",
                "Consider diabetes screening",
                "Regular exercise"
            ]
        else:
            interpretation = "Fasting blood sugar is normal."
            recommendations = ["Maintain healthy diet", "Regular monitoring"]
    
    elif param_name == 'restecg':
        if value == 0:
            status = "Normal"
            interpretation = "Normal ECG findings."
            recommendations = ["Continue regular check-ups"]
        elif value == 1:
            status = "ST-T Abnormality"
            interpretation = "ST-T wave abnormalities may indicate ventricular hypertrophy or ischemia."
            recommendations = ["Cardiac evaluation recommended", "Follow-up with cardiologist"]
        elif value == 2:
            status = "LV Hypertrophy"
            interpretation = "Left ventricular hypertrophy indicates heart enlargement, a risk factor."
            recommendations = ["Blood pressure monitoring", "Cardiac evaluation", "Echocardiogram may be indicated"]
    
    elif param_name == 'exang':
        if value == 1:
            status = "Present"
            interpretation = "Exercise-induced angina strongly suggests coronary artery disease."
            recommendations = ["Urgent cardiac evaluation", "Consider stress test or angiography", "Discuss treatment options"]
        else:
            interpretation = "No exercise-induced angina reported."
            recommendations = ["Continue regular exercise with appropriate precautions"]
    
    elif param_name == 'slope':
        if value == 0:
            status = "Upsloping"
            interpretation = "Upsloping ST segment is generally a normal finding."
            recommendations = ["Continue regular cardiac monitoring"]
        elif value == 1:
            status = "Flat"
            interpretation = "Flat ST segment may indicate ischemia."
            recommendations = ["Cardiac evaluation recommended", "Follow-up with cardiologist"]
        elif value == 2:
            status = "Downsloping"
            interpretation = "Downsloping ST segment strongly suggests myocardial ischemia."
            recommendations = ["Urgent cardiac evaluation", "Consider stress test or angiography"]
    
    elif param_name == 'ca':
        if value > 0:
            status = f"{value} Vessels"
            interpretation = f"Presence of {value} colored major vessels indicates coronary disease."
            recommendations = ["Comprehensive cardiac evaluation", "Discuss treatment options with cardiologist"]
        else:
            interpretation = "No major vessels colored by fluoroscopy."
            recommendations = ["Continue preventive measures"]
    
    elif param_name == 'thal':
        if value == 0:
            status = "Null"
            interpretation = "Null thalassemia value."
            recommendations = ["Additional testing may be needed"]
        elif value == 1:
            status = "Fixed Defect"
            interpretation = "Fixed defect indicates previous myocardial infarction."
            recommendations = ["Cardiac rehabilitation", "Secondary prevention measures", "Regular follow-up"]
        elif value == 2:
            status = "Normal"
            interpretation = "Normal blood flow."
            recommendations = ["Continue preventive measures"]
        elif value == 3:
            status = "Reversible Defect"
            interpretation = "Reversible defect suggests current ischemia."
            recommendations = ["Urgent cardiac evaluation", "Consider angiography", "Discuss treatment options"]
    
    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_heart_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_heart_overall_recommendation(detailed_analyses: List[DetailedAnalysis], risk_level: str) -> str:
    high_risk_params = [
        analysis for analysis in detailed_analyses 
        if analysis.status in ["High", "Elevated", "Present", "Downsloping", "Reversible Defect"]
        or (analysis.parameter == "ca" and analysis.value > 0)
    ]
    
    recommendation_parts = []
    
    if risk_level == "High Risk" or len(high_risk_params) >= 2:
        recommendation_parts.append("URGENT: Schedule an appointment with a cardiologist as soon as possible for comprehensive evaluation.")
    elif risk_level == "Moderate Risk" or len(high_risk_params) == 1:
        recommendation_parts.append("IMPORTANT: Follow up with your healthcare provider within the next 2 weeks to discuss these findings.")
    else:
        recommendation_parts.append("Continue with regular cardiac check-ups and maintain heart-healthy lifestyle.")
    
    # Specific recommendations based on parameters
    if any(a.parameter == "trestbps" and a.status == "High" for a in detailed_analyses):
        recommendation_parts.append(
            "Blood Pressure Management: Reduce sodium intake, maintain healthy weight, and consider DASH diet. "
            "Monitor your blood pressure regularly."
        )
    
    if any(a.parameter == "chol" and a.status == "High" for a in detailed_analyses):
        recommendation_parts.append(
            "Cholesterol Management: Reduce saturated fat intake, increase fiber, consider plant sterols. "
            "Follow up with lipid panel in 3 months."
        )
    
    if any(a.parameter == "fbs" and a.status == "Elevated" for a in detailed_analyses):
        recommendation_parts.append(
            "Blood Sugar Management: Limit refined carbohydrates, increase physical activity, maintain healthy weight. "
            "Consider comprehensive diabetes screening."
        )
    
    if any(a.parameter == "exang" and a.status == "Present" for a in detailed_analyses) or \
       any(a.parameter == "cp" and a.status != "Asymptomatic" for a in detailed_analyses):
        recommendation_parts.append(
            "Symptom Management: Note triggers of chest pain and report any changes in pattern or severity. "
            "Discuss appropriate medications with your cardiologist."
        )
    
    # General heart health recommendations
    lifestyle_recommendations = [
        "Heart-Healthy Diet: Emphasize fruits, vegetables, whole grains, lean proteins, and healthy fats.",
        "Regular Exercise: Aim for at least 150 minutes of moderate activity per week.",
        "Stress Management: Practice relaxation techniques such as deep breathing, meditation, or yoga.",
        "Sleep Hygiene: Prioritize 7-8 hours of quality sleep each night.",
        "Avoid Tobacco: If you smoke, seek support to quit.",
        "Limit Alcohol: If you drink alcohol, do so in moderation."
    ]
    
    # Add appropriate lifestyle recommendations based on risk
    if risk_level == "High Risk":
        recommendation_parts.append("Lifestyle Changes (consult physician before implementing): " + lifestyle_recommendations[0])
    elif risk_level == "Moderate Risk":
        recommendation_parts.append("Lifestyle Recommendations: " + " ".join(lifestyle_recommendations[:3]))
    else:
        recommendation_parts.append("Lifestyle Maintenance: " + " ".join(lifestyle_recommendations))
    
    monitoring = "Schedule follow-up assessments to monitor your cardiovascular health."
    recommendation_parts.append(monitoring)
    
    return " ".join(recommendation_parts)

# -------------------- Canser Models --------------------
class CancerInput(BaseModel):
    Age: int = Field(..., description="Age of the patient")
    Gender: int = Field(..., description="Gender (1-2)")
    Air_Pollution: int = Field(..., description="Air Pollution exposure level (1-8)")
    Alcohol_use: int = Field(..., description="Alcohol consumption level (1-8)")
    Dust_Allergy: int = Field(..., description="Dust Allergy level (1-8)")
    OccuPational_Hazards: int = Field(..., description="Occupational Hazards exposure (1-8)")  
    Genetic_Risk: int = Field(..., description="Genetic Risk level (1-7)")
    chronic_Lung_Disease: int = Field(..., description="Chronic Lung Disease level (1-7)")
    Balanced_Diet: int = Field(..., description="Balanced Diet level (1-7)")
    Obesity: int = Field(..., description="Obesity level (1-7)")
    Smoking: int = Field(..., description="Smoking level (1-8)")
    Passive_Smoker: int = Field(..., description="Passive Smoker exposure (1-8)")
    Chest_Pain: int = Field(..., description="Chest Pain level (1-9)")
    Coughing_of_Blood: int = Field(..., description="Coughing of Blood level (1-9)")
    Fatigue: int = Field(..., description="Fatigue level (1-9)")
    Weight_Loss: int = Field(..., description="Weight Loss level (1-8)")
    Shortness_of_Breath: int = Field(..., description="Shortness of Breath level (1-9)")
    Wheezing: int = Field(..., description="Wheezing level (1-8)")
    Swallowing_Difficulty: int = Field(..., description="Swallowing Difficulty level (1-7)")
    Clubbing_of_Finger_Nails: int = Field(..., description="Clubbing of Finger Nails level (1-9)")
    Frequent_Cold: int = Field(..., description="Frequent Cold occurrences (1-7)")
    Dry_Cough: int = Field(..., description="Dry Cough level (1-7)")
    Snoring: int = Field(..., description="Snoring level (1-7)")
     
    @model_validator(mode='after')
    def validate_fields(self) -> 'CancerInput':
        # Validate age
        if self.Age <= 0 or self.Age > 120:
            raise ValueError('Age must be between 1 and 120')
            
        # Validate gender
        if self.Gender not in [1, 2]:
            raise ValueError('Gender must be 1 or 2')
            
        # Validate ranges for risk factors (1-8)
        risk_factors_1_8 = {
            'Air_Pollution': self.Air_Pollution,
            'Alcohol_use': self.Alcohol_use,
            'Dust_Allergy': self.Dust_Allergy,
            'OccuPational_Hazards': self.OccuPational_Hazards,
            'Smoking': self.Smoking,
            'Passive_Smoker': self.Passive_Smoker,
            'Weight_Loss': self.Weight_Loss,
            'Wheezing': self.Wheezing
        }
        
        for field_name, value in risk_factors_1_8.items():
            if value < 1 or value > 8:
                raise ValueError(f'{field_name} must be between 1 and 8')
        
        # Validate ranges for risk factors (1-7)
        risk_factors_1_7 = {
            'Genetic_Risk': self.Genetic_Risk,
            'chronic_Lung_Disease': self.chronic_Lung_Disease,
            'Balanced_Diet': self.Balanced_Diet,
            'Obesity': self.Obesity,
            'Swallowing_Difficulty': self.Swallowing_Difficulty,
            'Frequent_Cold': self.Frequent_Cold,
            'Dry_Cough': self.Dry_Cough,
            'Snoring': self.Snoring
        }
        
        for field_name, value in risk_factors_1_7.items():
            if value < 1 or value > 7:
                raise ValueError(f'{field_name} must be between 1 and 7')
        
        # Validate ranges for symptoms (1-9)
        symptoms_1_9 = {
            'Chest_Pain': self.Chest_Pain,
            'Coughing_of_Blood': self.Coughing_of_Blood,
            'Fatigue': self.Fatigue,
            'Shortness_of_Breath': self.Shortness_of_Breath,
            'Clubbing_of_Finger_Nails': self.Clubbing_of_Finger_Nails
        }
        
        for field_name, value in symptoms_1_9.items():
            if value < 1 or value > 9:
                raise ValueError(f'{field_name} must be between 1 and 9')
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 55,
                "Gender": 1,
                "Air_Pollution": 5,
                "Alcohol_use": 3,
                "Dust_Allergy": 4,
                "OccuPational_Hazards": 4,
                "Genetic_Risk": 4,
                "chronic_Lung_Disease": 2,
                "Balanced_Diet": 3,
                "Obesity": 3,
                "Smoking": 6,
                "Passive_Smoker": 4,
                "Chest_Pain": 5,
                "Coughing_of_Blood": 3,
                "Fatigue": 4,
                "Weight_Loss": 4,
                "Shortness_of_Breath": 5,
                "Wheezing": 4,
                "Swallowing_Difficulty": 2,
                "Clubbing_of_Finger_Nails": 3,
                "Frequent_Cold": 4,
                "Dry_Cough": 5,
                "Snoring": 3
            }
        }
    }

class CancerPrediction(BaseModel):
    prediction: int
    probability: Dict[str, float]
    risk_level: str
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]

def analyze_canser_parameter(param_name: str, value: int) -> DetailedAnalysis:
    """Analyze a single parameter and provide detailed analysis"""

    # Define parameter thresholds and analysis
    analysis_map = {
        "Age": {
            "thresholds": [30, 50, 70],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Age is not a major factor at this point.",
                "Age is starting to become a risk factor.",
                "Age is a significant risk factor.",
                "Age is a major risk factor."
            ],
            "recommendations": [
                ["Regular check-ups are recommended even at younger ages"],
                ["Annual cancer screenings are recommended", "Maintain healthy lifestyle habits"],
                ["More frequent screenings are advised", "Consult with specialists for preventative measures"],
                ["Close monitoring is necessary", "Comprehensive screening program is recommended"]
            ]
        },
        "Air_Pollution": {
            "thresholds": [3, 5, 7],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Low air pollution exposure.",
                "Moderate air pollution exposure can contribute to cancer risk.",
                "High air pollution exposure is a significant concern.",
                "Very high air pollution exposure presents a serious risk factor."
            ],
            "recommendations": [
                ["Continue to avoid areas with high pollution"],
                ["Consider air purifiers for home and office", "Reduce outdoor activities during high pollution days"],
                ["Use appropriate masks when outdoors in polluted areas", "Monitor air quality daily"],
                ["Consider relocating to an area with better air quality", "Use HEPA filters in your home"]
            ]
        },
        "Alcohol_use": {
            "thresholds": [3, 5, 7],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Low alcohol consumption.",
                "Moderate alcohol consumption can increase risk.",
                "High alcohol consumption is a significant risk factor.",
                "Very high alcohol consumption substantially increases cancer risk."
            ],
            "recommendations": [
                ["Maintain current limited alcohol consumption"],
                ["Consider reducing alcohol intake", "Have alcohol-free days each week"],
                ["Seek support for reducing alcohol consumption", "Limit to no more than one drink per day"],
                ["Consider alcohol cessation program", "Consult healthcare provider about alcohol dependence"]
            ]
        },
        "Dust_Allergy": {
            "thresholds": [3, 5, 7],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Low sensitivity to dust allergens.",
                "Moderate dust allergy may cause respiratory issues.",
                "High dust allergy may contribute to chronic inflammation.",
                "Very high dust allergy can cause significant respiratory distress."
            ],
            "recommendations": [
                ["Regular cleaning to minimize dust"],
                ["Use allergen-proof bedding", "Consider air purifiers"],
                ["Consult with an allergist", "Use HEPA filters"],
                ["Medication management for allergies", "Consider immunotherapy"]
            ]
        },
        "OccuPational_Hazards": {
            "thresholds": [3, 5, 7],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Low occupational hazard exposure.",
                "Moderate occupational exposure to carcinogens.",
                "High occupational hazard exposure is concerning.",
                "Very high occupational exposure significantly increases risk."
            ],
            "recommendations": [
                ["Continue using proper safety equipment"],
                ["Ensure proper ventilation in workplace", "Use all recommended protective equipment"],
                ["Request workplace hazard assessment", "Consider job rotation to reduce exposure"],
                ["Seek alternative position with less exposure", "Regular medical monitoring for early detection"]
            ]
        },
        "Genetic_Risk": {
            "thresholds": [2, 4, 6],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Low genetic predisposition to cancer.",
                "Moderate genetic risk factors present.",
                "High genetic risk for certain cancer types.",
                "Very high genetic risk requires careful monitoring."
            ],
            "recommendations": [
                ["Standard cancer screenings appropriate for age"],
                ["Consider genetic counseling", "Earlier screening may be appropriate"],
                ["Genetic testing recommended", "Customized screening schedule"],
                ["Comprehensive genetic counseling", "Consider preventative interventions"]
            ]
        },
        "chronic_Lung_Disease": {
            "thresholds": [2, 4, 6],
            "levels": ["Low", "Moderate", "High", "Very High"],
            "interpretations": [
                "Minimal lung disease present.",
                "Moderate lung disease can increase cancer risk.",
                "Significant lung disease is a major risk factor.",
                "Severe lung disease substantially increases lung cancer risk."
            ],
            "recommendations": [
                ["Regular pulmonary function tests"],
                ["Pulmonary rehabilitation may be beneficial", "Avoid respiratory irritants"],
                ["Regular specialist monitoring", "Medication optimization"],
                ["Comprehensive pulmonary care", "Consider oxygen therapy assessment"]
            ]
        },
        "Balanced_Diet": {
            "thresholds": [2, 4, 6],
            "levels": ["Very Poor", "Poor", "Adequate", "Good"],
            "interpretations": [
                "Very poor nutritional balance increases cancer risk.",
                "Poor diet lacking in protective nutrients.",
                "Adequate diet provides some cancer-fighting nutrients.",
                "Good nutritional balance helps reduce cancer risk."
            ],
            "recommendations": [
                ["Urgent dietary improvements needed", "Consult with a nutritionist"],
                ["Increase fruits and vegetables", "Reduce processed foods"],
                ["Continue healthy eating habits", "Consider more plant-based options"],
                ["Maintain excellent dietary habits", "Consider periodic nutritional assessments"]
            ]
        },
        "Obesity": {
            "thresholds": [2, 4, 6],
            "levels": ["Normal weight", "Overweight", "Obese", "Severely obese"],
            "interpretations": [
                "Weight is in a healthy range.",
                "Overweight status slightly increases cancer risk.",
                "Obesity significantly increases risk for several cancers.",
                "Severe obesity is a major risk factor."
            ],
            "recommendations": [
                ["Maintain healthy weight through diet and exercise"],
                ["Aim for 5-10% weight reduction", "Increase physical activity"],
                ["Structured weight loss program recommended", "Regular monitoring of metabolic factors"],
                ["Medical weight management advised", "Consider bariatric surgery consultation"]
            ]
        },
        "Smoking": {
            "thresholds": [2, 4, 7],
            "levels": ["Non-smoker", "Light smoker", "Moderate smoker", "Heavy smoker"],
            "interpretations": [
                "Minimal to no smoking history.",
                "Light smoking still carries significant cancer risk.",
                "Moderate smoking substantially increases cancer risk.",
                "Heavy smoking is a major cancer risk factor."
            ],
            "recommendations": [
                ["Maintain smoke-free lifestyle"],
                ["Smoking cessation strongly recommended", "Consider nicotine replacement therapy"],
                ["Urgent smoking cessation advised", "Consider prescription medications to help quit"],
                ["Immediate cessation critical", "Comprehensive smoking cessation program recommended"]
            ]
        },
        "Passive_Smoker": {
            "thresholds": [2, 4, 7],
            "levels": ["Minimal exposure", "Moderate exposure", "High exposure", "Very high exposure"],
            "interpretations": [
                "Minimal secondhand smoke exposure.",
                "Moderate passive smoke exposure increases risk.",
                "High secondhand smoke exposure is concerning.",
                "Very high passive smoke exposure significantly increases risk."
            ],
            "recommendations": [
                ["Continue avoiding secondhand smoke"],
                ["Create smoke-free environments", "Avoid areas where smoking is permitted"],
                ["Take active measures to reduce exposure", "Air purifiers in shared spaces"],
                ["Consider relocating if in a high-exposure living situation", "Establish strict no-smoking policies"]
            ]
        },
        "Chest_Pain": {
            "thresholds": [3, 5, 7],
            "levels": ["Minimal", "Moderate", "Severe", "Very severe"],
            "interpretations": [
                "Minimal chest pain reported.",
                "Moderate chest pain requires evaluation.",
                "Significant chest pain is concerning.",
                "Severe chest pain needs immediate medical attention."
            ],
            "recommendations": [
                ["Monitor any changes in frequency or intensity"],
                ["Medical evaluation recommended", "Keep a symptom diary"],
                ["Urgent medical assessment needed", "Thoracic imaging may be required"],
                ["Immediate medical attention required", "Comprehensive cancer screening advised"]
            ]
        },
        "Coughing_of_Blood": {
            "thresholds": [2, 4, 6],
            "levels": ["None/Minimal", "Occasional", "Frequent", "Severe"],
            "interpretations": [
                "No significant hemoptysis noted.",
                "Occasional blood in cough requires evaluation.",
                "Frequent hemoptysis is a serious symptom.",
                "Severe hemoptysis requires urgent care."
            ],
            "recommendations": [
                ["Report any future instances to healthcare provider"],
                ["Medical evaluation recommended", "Chest imaging advised"],
                ["Urgent specialist referral needed", "Bronchoscopy may be required"],
                ["Emergency medical attention required", "Hospital admission may be necessary"]
            ]
        },
        "Fatigue": {
            "thresholds": [3, 5, 7],
            "levels": ["Mild", "Moderate", "Severe", "Extreme"],
            "interpretations": [
                "Mild fatigue within normal limits.",
                "Moderate fatigue affecting daily activities.",
                "Severe fatigue is concerning and needs evaluation.",
                "Extreme fatigue may indicate serious underlying condition."
            ],
            "recommendations": [
                ["Ensure adequate sleep and rest"],
                ["Medical evaluation recommended", "Consider anemia screening"],
                ["Comprehensive medical workup needed", "Energy conservation strategies"],
                ["Urgent medical assessment required", "Consider cancer-related fatigue evaluation"]
            ]
        },
        "Weight_Loss": {
            "thresholds": [3, 5, 7],
            "levels": ["Minimal", "Moderate", "Significant", "Severe"],
            "interpretations": [
                "Minimal unintentional weight loss.",
                "Moderate weight loss should be evaluated.",
                "Significant unintentional weight loss is concerning.",
                "Severe weight loss requires urgent evaluation."
            ],
            "recommendations": [
                ["Monitor weight regularly"],
                ["Medical evaluation recommended", "Nutritional assessment"],
                ["Urgent medical workup needed", "High-calorie nutritional support"],
                ["Immediate specialist referral", "Comprehensive cancer screening"]
            ]
        },
        "Shortness_of_Breath": {
            "thresholds": [3, 5, 7],
            "levels": ["Mild", "Moderate", "Severe", "Extreme"],
            "interpretations": [
                "Mild shortness of breath with exertion.",
                "Moderate dyspnea affecting daily activities.",
                "Severe breathing difficulty is very concerning.",
                "Extreme breathing difficulty requires urgent care."
            ],
            "recommendations": [
                ["Monitor for changes in breathing capacity"],
                ["Pulmonary function testing recommended", "Consider cardiac evaluation"],
                ["Urgent pulmonology consultation", "Oxygen assessment needed"],
                ["Emergency medical attention required", "Comprehensive pulmonary and cardiac workup"]
            ]
        },
        "Wheezing": {
            "thresholds": [3, 5, 7],
            "levels": ["Occasional", "Intermittent", "Frequent", "Constant"],
            "interpretations": [
                "Occasional wheezing noted.",
                "Intermittent wheezing may indicate airway issues.",
                "Frequent wheezing suggests significant airway obstruction.",
                "Constant wheezing indicates severe airway compromise."
            ],
            "recommendations": [
                ["Track frequency and triggers"],
                ["Pulmonary function testing advised", "Consider inhaler therapy"],
                ["Pulmonologist consultation needed", "Optimized respiratory medications"],
                ["Urgent specialist care required", "Imaging to rule out obstructive mass"]
            ]
        },
        "Swallowing_Difficulty": {
            "thresholds": [2, 4, 6],
            "levels": ["Minimal", "Moderate", "Severe", "Extreme"],
            "interpretations": [
                "Minimal swallowing difficulties.",
                "Moderate dysphagia requires evaluation.",
                "Severe swallowing difficulty is concerning.",
                "Extreme dysphagia requires urgent care."
            ],
            "recommendations": [
                ["Monitor for worsening symptoms"],
                ["ENT evaluation recommended", "Modified barium swallow study"],
                ["Urgent specialist referral", "Nutritional intervention needed"],
                ["Immediate medical attention", "Upper endoscopy recommended"]
            ]
        },
        "Clubbing_of_Finger_Nails": {
            "thresholds": [3, 5, 7],
            "levels": ["None/Minimal", "Early signs", "Moderate", "Severe"],
            "interpretations": [
                "No significant clubbing observed.",
                "Early nail clubbing may indicate cardiopulmonary issues.",
                "Moderate clubbing is associated with lung disease.",
                "Severe clubbing strongly associated with lung cancer."
            ],
            "recommendations": [
                ["Monitor for any changes in nail appearance"],
                ["Pulmonary and cardiac evaluation recommended"],
                ["Chest imaging advised", "Comprehensive cardiopulmonary assessment"],
                ["Urgent oncology referral", "Lung cancer screening recommended"]
            ]
        },
        "Frequent_Cold": {
            "thresholds": [2, 4, 6],
            "levels": ["Occasional", "Frequent", "Very frequent", "Constant"],
            "interpretations": [
                "Normal frequency of respiratory infections.",
                "Increased susceptibility to respiratory infections.",
                "Very frequent infections suggest immune dysfunction.",
                "Constant infections indicate significant immune compromise."
            ],
            "recommendations": [
                ["Practice good hand hygiene"],
                ["Immune system evaluation recommended", "Consider vitamin D testing"],
                ["Immunology consultation advised", "Comprehensive immune workup"],
                ["Specialist evaluation for immune deficiency", "Cancer screening recommended"]
            ]
        },
        "Dry_Cough": {
            "thresholds": [2, 4, 6],
            "levels": ["Occasional", "Frequent", "Persistent", "Severe persistent"],
            "interpretations": [
                "Occasional dry cough not concerning.",
                "Frequent dry cough requires evaluation.",
                "Persistent dry cough is a concerning symptom.",
                "Severe persistent cough requires thorough investigation."
            ],
            "recommendations": [
                ["Monitor for changes in frequency or character"],
                ["Medical evaluation recommended", "Consider allergy testing"],
                ["Chest imaging advised", "Pulmonary function testing"],
                ["Urgent specialist referral", "Comprehensive cancer screening"]
            ]
        },
        "Snoring": {
            "thresholds": [2, 4, 6],
            "levels": ["Minimal", "Moderate", "Severe", "Very severe"],
            "interpretations": [
                "Minimal snoring not concerning.",
                "Moderate snoring may indicate partial airway obstruction.",
                "Severe snoring suggests significant airway issues.",
                "Very severe snoring with possible sleep apnea."
            ],
            "recommendations": [
                ["Monitor for changes in sleep quality"],
                ["Consider sleep position changes", "Weight management if appropriate"],
                ["Sleep study recommended", "ENT evaluation"],
                ["Urgent sleep medicine consultation", "CPAP therapy may be needed"]
            ]
        }
    }
        
     
    # Get analysis for specific parameter
    param_analysis = analysis_map.get(param_name, {
        "thresholds": [3, 5, 7],
        "levels": ["Low", "Moderate", "High", "Very High"],
        "interpretations": [
            "Parameter is at a low level.",
            "Parameter is at a moderate level.",
            "Parameter is at a high level.",
            "Parameter is at a very high level."
        ],
        "recommendations": [
            ["Continue current management"],
            ["Consider lifestyle modifications"],
            ["Medical evaluation recommended"],
            ["Urgent medical attention advised"]
        ]
    })

    # Determine risk level based on thresholds
    thresholds = param_analysis["thresholds"]
    levels = param_analysis["levels"]
    interpretations = param_analysis["interpretations"]
    all_recommendations = param_analysis["recommendations"]
    
    level_index = 0
    for threshold in thresholds:
        if value > threshold:
            level_index += 1
        else:
            break
    
    risk_level = levels[level_index]
    interpretation = interpretations[level_index]
    recommendations = all_recommendations[level_index]
    
    
    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=risk_level,  # Set status equal to risk_level for consistency
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_canser_risk_level(prediction: int) -> str:
    """Map prediction level to a descriptive risk level"""
    risk_levels = {
        1: "Low Risk",
        2: "Moderate Risk",
        3: "High Risk"
    }
    return risk_levels.get(prediction, "Unknown Risk")


def generate_canser_overall_recommendation(detailed_analyses: List[DetailedAnalysis], prediction: int) -> str:
    """Generate overall recommendation based on analyses and prediction"""
    risk_level = get_canser_risk_level(prediction)
    
    # Identify high-risk parameters
    high_risk_params = [
        analysis for analysis in detailed_analyses 
        if analysis.status in ["High", "Very High", "Severe", "Extreme", "Very severe", "Persistent", "Severe persistent"]
    ]
    
    # Identify lifestyle factors that need intervention
    lifestyle_factors = [
        analysis for analysis in detailed_analyses 
        if analysis.parameter in ["Smoking", "Alcohol_use", "Balanced_Diet", "Obesity", "Air_Pollution", "Passive_Smoker"]
        and analysis.status in ["High", "Very High", "Poor", "Very Poor", "Obese", "Severely obese", "Moderate smoker", "Heavy smoker"]
    ]
    
    # Identify concerning symptoms
    concerning_symptoms = [
        analysis for analysis in detailed_analyses 
        if analysis.parameter in ["Chest_Pain", "Coughing_of_Blood", "Weight_Loss", "Shortness_of_Breath", 
                                 "Wheezing", "Clubbing_of_Finger_Nails", "Dry_Cough", "Swallowing_Difficulty"]
        and analysis.status in ["Moderate", "Severe", "Frequent", "Persistent", "Very severe", "Extreme"]
    ]
    
    recommendation_parts = []
    
    # Overall risk assessment
    if prediction == 1:
        recommendation_parts.append(
            "Your assessment suggests a LOW RISK of lung cancer. However, regular monitoring and preventive measures are still recommended."
        )
    elif prediction == 2:
        recommendation_parts.append(
            "Your assessment suggests a MODERATE RISK of lung cancer. Targeted lifestyle modifications and regular screening are recommended."
        )
    elif prediction == 3:
        recommendation_parts.append(
            "Your assessment suggests a HIGH RISK of lung cancer. Urgent medical consultation and comprehensive screening are strongly advised."
        )
    
    # Addressing symptoms
    if concerning_symptoms:
        if prediction == 3 or len(concerning_symptoms) >= 3:
            recommendation_parts.append(
                "URGENT: Schedule a comprehensive evaluation with a pulmonologist or oncologist within the next 1-2 weeks."
            )
        else:
            recommendation_parts.append(
                "IMPORTANT: Schedule a medical evaluation for your respiratory symptoms within the next month."
            )
    
    # Lifestyle recommendations
    if lifestyle_factors:
        lifestyle_changes = []
        
        for factor in lifestyle_factors:
            if factor.parameter == "Smoking" and factor.status in ["Moderate smoker", "Heavy smoker"]:
                lifestyle_changes.append("Smoking cessation is critical - consider a structured cessation program")
            
            elif factor.parameter == "Alcohol_use" and factor.status in ["High", "Very High"]:
                lifestyle_changes.append("Reduce alcohol consumption to no more than one drink per day")
            
            elif factor.parameter == "Balanced_Diet" and factor.status in ["Poor", "Very Poor"]:
                lifestyle_changes.append("Improve your diet with more vegetables, fruits, and whole grains")
            
            elif factor.parameter == "Obesity" and factor.status in ["Obese", "Severely obese"]:
                lifestyle_changes.append("Weight management through diet and exercise")
            
            elif factor.parameter == "Air_Pollution" and factor.status in ["High", "Very High"]:
                lifestyle_changes.append("Minimize exposure to air pollution using masks and air purifiers")
            
            elif factor.parameter == "Passive_Smoker" and factor.status in ["High exposure", "Very high exposure"]:
                lifestyle_changes.append("Avoid secondhand smoke completely")
        
        if lifestyle_changes:
            recommendation_parts.append("Lifestyle Modifications: " + ". ".join(lifestyle_changes) + ".")
    
    # Screening recommendations
    if prediction == 3 or (prediction == 2 and len(high_risk_params) >= 3):
        recommendation_parts.append(
            "Screening Recommendations: Low-dose CT scan of the chest is advised. Discuss with your healthcare provider about the appropriate screening interval."
        )
    elif prediction == 2:
        recommendation_parts.append(
            "Screening Recommendations: Consider baseline chest imaging and pulmonary function tests. Discuss screening intervals with your doctor."
        )
    else:
        recommendation_parts.append(
            "Screening Recommendations: Follow age-appropriate cancer screening guidelines. Consult your doctor if symptoms develop or worsen."
        )
    
    # Follow-up recommendation
    if prediction == 3:
        recommendation_parts.append(
            "Follow-up: Regular monitoring every 3-6 months with your healthcare provider is strongly recommended."
        )
    elif prediction == 2:
        recommendation_parts.append(
            "Follow-up: Schedule a follow-up appointment in 6 months to reassess your risk factors and symptoms."
        )
    else:
        recommendation_parts.append(
            "Follow-up: Annual health checkups are recommended to monitor any changes in your risk profile."
        )
    
    return " ".join(recommendation_parts)

# -------------------- Model Manager --------------------
class ModelManager:
    def __init__(self):
        self.diabetes_model = None
        self.diabetes_scaler = None
        self.anemia_model = None
        self.anemia_scaler = None
        self.hepc_model = None
        self.hepc_scaler = None
        self.kidney_model = None  
        self.kidney_scaler = None  
        self.heart_model = None
        self.heart_scaler = None
        self.canser_model = None
        self.canser_scaler = None
        self.load_models()


    def load_models(self):
        try:
            # Load diabetes models
            if not DIABETES_MODEL_PATH.exists():
                raise FileNotFoundError(f"Diabetes model file not found at {DIABETES_MODEL_PATH}")
            if not DIABETES_SCALER_PATH.exists():
                raise FileNotFoundError(f"Diabetes scaler file not found at {DIABETES_SCALER_PATH}")

            self.diabetes_model = joblib.load(DIABETES_MODEL_PATH)
            self.diabetes_scaler = joblib.load(DIABETES_SCALER_PATH)
            logger.info("Diabetes models loaded successfully")

            # Load anemia models
            if not ANEMIA_MODEL_PATH.exists():
                raise FileNotFoundError(f"Anemia model file not found at {ANEMIA_MODEL_PATH}")
            if not ANEMIA_SCALER_PATH.exists():
                raise FileNotFoundError(f"Anemia scaler file not found at {ANEMIA_SCALER_PATH}")

            self.anemia_model = joblib.load(ANEMIA_MODEL_PATH)
            self.anemia_scaler = joblib.load(ANEMIA_SCALER_PATH)
            logger.info("Anemia models loaded successfully")

            # Load hepc models
            if not HEPC_MODEL_PATH.exists():
                raise FileNotFoundError(f"Hepatitis c model file not found at {HEPC_MODEL_PATH}")
            if not HEPC_SCALER_PATH.exists():
                raise FileNotFoundError(f"Hepatitis c scaler file not found at {HEPC_SCALER_PATH}")

            self.hepc_model = joblib.load(HEPC_MODEL_PATH)
            self.hepc_scaler = joblib.load(HEPC_SCALER_PATH)
            logger.info("HepC models loaded successfully")

            
            # Load kidney models
            if not KIDNEY_MODEL_PATH.exists():
                raise FileNotFoundError(f"Kidney model file not found at {KIDNEY_MODEL_PATH}")
            if not KIDNEY_SCALER_PATH.exists():
                raise FileNotFoundError(f"Kidney scaler file not found at {KIDNEY_SCALER_PATH}")

            self.kidney_model = joblib.load(KIDNEY_MODEL_PATH)
            self.kidney_scaler = joblib.load(KIDNEY_SCALER_PATH)
            logger.info("Kidney models loaded successfully")

            # Load heart models
            if not HEART_MODEL_PATH.exists():
                raise FileNotFoundError(f"Heart Model file not found at {HEART_MODEL_PATH}")
            if not HEART_SCALER_PATH.exists():
                raise FileNotFoundError(f"Heart Scaler file not found at {HEART_SCALER_PATH}")
            
            self.heart_model = joblib.load(HEART_MODEL_PATH)
            self.heart_scaler = joblib.load(HEART_SCALER_PATH)
            logger.info("Heart Models loaded successfully")
            
            # Load canser models
            if not CANSER_MODEL_PATH.exists():
                raise FileNotFoundError(f"Canser Model file not found at {CANSER_MODEL_PATH}")
            if not CANSER_SCALER_PATH.exists():
                raise FileNotFoundError(f"canser Scaler file not found at {CANSER_SCALER_PATH}")
            
            self.canser_model = joblib.load(CANSER_MODEL_PATH)
            self.canser_scaler = joblib.load(CANSER_SCALER_PATH)
            logger.info("Canser Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
# -------------------- FastAPI Application --------------------
app = FastAPI(
title="Health Prediction API",
description="API for predicting diabetes , anemia , hepatitis c , kidney , heart, canser disease risk using trained machine learning models",
version="1.0.0"
)

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = ModelManager()
    logger.info("Model manager initialized during startup")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Health Prediction API",
        "diabetes_model_status": "loaded" if model_manager and model_manager.diabetes_model is not None else "not loaded",
        "anemia_model_status": "loaded" if model_manager and model_manager.anemia_model is not None else "not loaded",
        "hepc_model_status": "loaded" if model_manager and model_manager.hepc_model is not None else "not loaded",
        "kidney_model_status": "loaded" if model_manager and model_manager.kidney_model is not None else "not loaded",
        "heart_model_status": "loaded" if model_manager and model_manager.heart_model is not None else "not loaded",
        "canser_model_status": "loaded" if model_manager and model_manager.canser_model is not None else "not loaded",
        "documentation": "/docs",
        "endpoints": {
            "/predict/diabetes": "POST - Make diabetes predictions",
            "/predict/anemia": "POST - Make anemia predictions",
            "/predict/hepc": "POST - Make Hepatitis predictions",
            "/predict/kidney": "POST - Make Kidney disease predictions",
            "/predict/heart": "POST - Make heart disease predictions",
            "/predict/canser": "POST - Make canser disease predictions",
            "/health": "GET - Check API health"
        }
    }
@app.get("/health")
async def health_check():
    if not model_manager or model_manager.diabetes_model is None or model_manager.diabetes_scaler is None:
        raise HTTPException(status_code=503, detail="Diabetes model or scaler not loaded")

    if not model_manager or model_manager.anemia_model is None or model_manager.anemia_scaler is None:
        raise HTTPException(status_code=503, detail="Anemia model or scaler not loaded")

    if not model_manager or model_manager.hepc_model is None or model_manager.hepc_scaler is None:
        raise HTTPException(status_code=503, detail="HepC model or scaler not loaded")
    
    if not model_manager or model_manager.kidney_model is None or model_manager.kidney_scaler is None:
        raise HTTPException(status_code=503, detail="Kidney model or scaler not loaded")
        
    if not model_manager or model_manager.heart_model is None or model_manager.heart_scaler is None:
        raise HTTPException(status_code=503, detail="Heart model or scaler not loaded")
            
    if not model_manager or model_manager.canser_model is None or model_manager.canser_scaler is None:
        raise HTTPException(status_code=503, detail="Canser model or scaler not loaded")

    diabetes_features = []
    if hasattr(model_manager.diabetes_scaler, 'feature_names_in_'):
        diabetes_features = model_manager.diabetes_scaler.feature_names_in_.tolist()

    anemia_features = []
    if hasattr(model_manager.anemia_scaler, 'feature_names_in_'):
        anemia_features = model_manager.anemia_scaler.feature_names_in_.tolist()

    hepc_features = []
    if hasattr(model_manager.hepc_scaler, 'feature_names_in_'):
        hepc_features = model_manager.hepc_scaler.feature_names_in_.tolist()

    kidney_features = []
    if hasattr(model_manager.kidney_scaler, 'feature_names_in_'):
        kidney_features = model_manager.kidney_scaler.feature_names_in_.tolist()

    heart_features = []
    if hasattr(model_manager.kidney_scaler, 'feature_names_in_'):
        heart_features = model_manager.heart_scaler.feature_names_in_.tolist()
 
    canser_features = []
    if hasattr(model_manager.canser_scaler, 'feature_names_in_'):
        canser_features = model_manager.canser_scaler.feature_names_in_.tolist()

    return {
        "status": "healthy",
        "diabetes_model": {
            "loaded": True,
            "model_path": str(DIABETES_MODEL_PATH),
            "scaler_path": str(DIABETES_SCALER_PATH),
            "features": diabetes_features
        },
        "anemia_model": {
            "loaded": True,
            "model_path": str(ANEMIA_MODEL_PATH),
            "scaler_path": str(ANEMIA_SCALER_PATH),
            "features": anemia_features
        },
        "hepc_model": {
            "loaded": True,
            "model_path": str(HEPC_MODEL_PATH),
            "scaler_path": str(HEPC_SCALER_PATH),
            "features": hepc_features
        },
        "kidney_model": {
            "loaded": True,
            "model_path": str(KIDNEY_MODEL_PATH),
            "scaler_path": str(KIDNEY_SCALER_PATH),
            "features": kidney_features
        },
        "heart_model": {
            "loaded": True,
            "model_path": str(HEART_MODEL_PATH),
            "scaler_path": str(HEART_SCALER_PATH),
            "features": heart_features
        },
        "canser_model": {
            "loaded": True,
            "model_path": str(CANSER_MODEL_PATH),
            "scaler_path": str(CANSER_SCALER_PATH),
            "features": canser_features
        }
    }

@app.post("/predict/diabetes", response_model=DiabetesPrediction)
async def predict_diabetes(data: DiabetesInput):
    logger.info("Received diabetes prediction request")

    if not model_manager or model_manager.diabetes_model is None or model_manager.diabetes_scaler is None:
        raise HTTPException(status_code=503, detail="Diabetes model or scaler not loaded")

    try:
        features = np.array([[
            data.age,
            data.hypertension,
            data.bmi,
            data.hbA1c_level,
            data.blood_glucose_level
        ]])

        logger.debug(f"Original input data: {features}")
        features_scaled = model_manager.diabetes_scaler.transform(features)
        logger.debug(f"Scaled input data: {features_scaled}")

        prediction = model_manager.diabetes_model.predict(features_scaled)[0]

        if hasattr(model_manager.diabetes_model, "predict_proba"):
            probability = model_manager.diabetes_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.diabetes_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])

        risk_level = get_diabetes_risk_level(probability)
        recommendation = get_diabetes_recommendation(risk_level, data.age, data.bmi)

        response = DiabetesPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            recommendation=recommendation,
            input_values=data.model_dump()
        )

        logger.info(f"Diabetes prediction complete: {prediction}, Risk Level: {risk_level}")
        return response

    except Exception as e:
        logger.error(f"Diabetes prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/anemia", response_model=AnemiaPrediction)
async def predict_anemia(data: AnemiaInput):
    logger.info("Received anemia prediction request")

    if not model_manager or model_manager.anemia_model is None or model_manager.anemia_scaler is None:
        raise HTTPException(status_code=503, detail="Anemia model or scaler not loaded")

    try:
        features = np.array([[
            data.hemoglobin,
            data.mch,
            data.mchc,
            data.mcv
        ]])

        features_scaled = model_manager.anemia_scaler.transform(features)
        prediction = model_manager.anemia_model.predict(features_scaled)[0]

        if hasattr(model_manager.anemia_model, "predict_proba"):
            probability = model_manager.anemia_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.anemia_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])

        risk_level = get_anemia_risk_level(probability)

        # Perform detailed analysis of each parameter
        ranges = ParameterRanges()
        detailed_analyses = [
            analyze_parameter('hemoglobin', data.hemoglobin, ranges),
            analyze_parameter('mch', data.mch, ranges),
            analyze_parameter('mchc', data.mchc, ranges),
            analyze_parameter('mcv', data.mcv, ranges)
        ]

        overall_recommendation = generate_overall_recommendation(detailed_analyses, risk_level)

        response = AnemiaPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            detailed_analysis=detailed_analyses,
            overall_recommendation=overall_recommendation,
            input_values=data.model_dump()
        )

        logger.info(f"Anemia prediction complete: {prediction}, Risk Level: {risk_level}")
        return response

    except Exception as e:
        logger.error(f"Anemia prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/hepc", response_model=HepCPrediction)
async def predict_hepc(data: HepCInput):
    logger.info("Received hepatitis C prediction request")

    if not model_manager or model_manager.hepc_model is None or model_manager.hepc_scaler is None:
        raise HTTPException(status_code=503, detail="Hepatitis C model or scaler not loaded")

    try:
        features = np.array([[
            data.sex,
            data.age,
            data.alb,
            data.alp,
            data.alt,
            data.ast,
            data.bil,
            data.che,
            data.chol,
            data.crea,
            data.ggt,
            data.prot
        ]])

        features_scaled = model_manager.hepc_scaler.transform(features)
        prediction = model_manager.hepc_model.predict(features_scaled)[0]

        if hasattr(model_manager.hepc_model, "predict_proba"):
            probability = model_manager.hepc_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.hepc_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])

        risk_level = get_hepc_risk_level(probability)
        disease_stage = get_disease_stage(int(prediction))

        ranges = HepCParameterRanges()
        detailed_analyses = [
            analyze_hepc_parameter('alb', data.alb, ranges),
            analyze_hepc_parameter('alp', data.alp, ranges),
            analyze_hepc_parameter('alt', data.alt, ranges),
            analyze_hepc_parameter('ast', data.ast, ranges),
            analyze_hepc_parameter('bil', data.bil, ranges),
            analyze_hepc_parameter('che', data.che, ranges),
            analyze_hepc_parameter('chol', data.chol, ranges),
            analyze_hepc_parameter('crea', data.crea, ranges),
            analyze_hepc_parameter('ggt', data.ggt, ranges),
            analyze_hepc_parameter('prot', data.prot, ranges)
        ]

        stage_specific_recommendation = f"\n\nDisease Stage: {disease_stage['stage']}\n{disease_stage['description']}\n"
        overall_recommendation = generate_hepc_recommendation(detailed_analyses, risk_level) + stage_specific_recommendation

        response = HepCPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            disease_stage=disease_stage,
            detailed_analysis=detailed_analyses,
            overall_recommendation=overall_recommendation,
            input_values=data.model_dump()
        )

        logger.info(f"Hepatitis C prediction complete: {prediction}, Risk Level: {risk_level}, Disease Stage: {disease_stage['stage']}")
        return response

    except Exception as e:
        logger.error(f"Hepatitis C prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/predict/kidney", response_model=KidneyPrediction)
async def predict_kidney_disease(data: KidneyInput):
    """Predict kidney disease risk from clinical parameters"""
    try:
        # 1. Verify models are loaded
        if not hasattr(model_manager, 'kidney_model') or not hasattr(model_manager, 'kidney_scaler'):
            raise HTTPException(
                status_code=503,
                detail="Kidney prediction models not properly initialized"
            )
        
        if model_manager.kidney_model is None or model_manager.kidney_scaler is None:
            raise HTTPException(
                status_code=503,
                detail="Kidney prediction service unavailable (models not loaded)"
            )

        # 2. Prepare input features (13 total features)
        features = np.array([[
            data.specific_gravity,    # Numeric (1.010-1.025)
            data.albumin,             # Numeric (0-5 scale)
            data.red_blood_cells,     # Binary (0=normal, 1=abnormal)
            data.pus_cell,            # Binary (0=normal, 1=abnormal)
            data.blood_urea,          # Numeric (mg/dL)
            data.serum_creatinine,    # Numeric (mg/dL)
            data.sodium,              # Numeric (mEq/L)
            data.potassium,           # Numeric (mEq/L)
            data.hemoglobin,          # Numeric (g/dL)
            data.packed_cell_volume,  # Numeric (%)
            data.hypertension,        # Binary (0=no, 1=yes)
            data.diabetes_mellitus,   # Binary (0=no, 1=yes)
            data.anemia               # Binary (0=no, 1=yes)
        ]])

        # 3. Scale features and make prediction
        try:
            features_scaled = model_manager.kidney_scaler.transform(features)
            prediction = model_manager.kidney_model.predict(features_scaled)[0]
            
            # Handle probability prediction
            if hasattr(model_manager.kidney_model, "predict_proba"):
                probability = float(model_manager.kidney_model.predict_proba(features_scaled)[0][1])
            else:
                probability = float(prediction)  # Fallback
        except Exception as e:
            logger.error(f"Prediction processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction processing error: {str(e)}"
            )

        # 4. Generate detailed analysis for each parameter
        ranges = KidneyParameterRanges()
        detailed_analyses = []
        ranges = KidneyParameterRanges()
        numeric_fields = [
                "specific_gravity", "albumin", "blood_urea", "serum_creatinine",
                "sodium", "potassium", "hemoglobin", "packed_cell_volume"
        ]
        for field in numeric_fields:
                value = getattr(data, field)
                analysis = analyze_kidney_numeric_parameter(field, value, ranges)
                detailed_analyses.append(analysis)

        binary_fields = ["red_blood_cells", "pus_cell", "hypertension", "diabetes_mellitus", "anemia"]

        for field in binary_fields:
                value = getattr(data, field)
                analysis = analyze_kidney_categorical_parameter(field, value, ranges)
                detailed_analyses.append(analysis)

        # 5. Prepare response
        risk_level = get_kidney_risk_level(probability)
        return KidneyPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            detailed_analysis=detailed_analyses,
            overall_recommendation=generate_kidney_overall_recommendation(
                detailed_analyses,
                risk_level
            ),
            input_values=data.model_dump()
        )

    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except HTTPException:
        raise  # Re-raise already handled HTTP exceptions
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@app.post("/predict/heart", response_model=HeartDiseasePrediction)
async def predict_heart_disease(data: HeartDiseaseInput):
    logger.info("Received heart disease prediction request")
    
    if model_manager.heart_model is None or model_manager.heart_scaler is None:
        raise HTTPException(status_code=503, detail="Heart Model or Scaler not loaded")
    
    try:
        features = np.array([[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]])
        
        features_scaled = model_manager.heart_scaler.transform(features)
        prediction = model_manager.heart_model.predict(features_scaled)[0]
        
        if hasattr(model_manager.heart_model, "predict_proba"):
            probability = model_manager.heart_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.heart_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])
        
        risk_level = get_heart_risk_level(probability)
        
        # Perform detailed analysis of each parameter
        ranges = HeartParameterRanges()
        detailed_analyses = [
            analyze_heart_parameter('age', data.age, ranges),
            analyze_heart_parameter('sex', data.sex, ranges),
            analyze_heart_parameter('cp', data.cp, ranges),
            analyze_heart_parameter('trestbps', data.trestbps, ranges),
            analyze_heart_parameter('chol', data.chol, ranges),
            analyze_heart_parameter('fbs', data.fbs, ranges),
            analyze_heart_parameter('restecg', data.restecg, ranges),
            analyze_heart_parameter('thalach', data.thalach, ranges, data.age),
            analyze_heart_parameter('exang', data.exang, ranges),
            analyze_heart_parameter('oldpeak', data.oldpeak, ranges),
            analyze_heart_parameter('slope', data.slope, ranges),
            analyze_heart_parameter('ca', data.ca, ranges),
            analyze_heart_parameter('thal', data.thal, ranges)
        ]
        
        overall_recommendation = generate_heart_overall_recommendation(detailed_analyses, risk_level)
        
        response = HeartDiseasePrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            detailed_analysis=detailed_analyses,
            overall_recommendation=overall_recommendation,
            input_values=data.model_dump()
        )
        
        logger.info(f"Heart disease prediction complete: {prediction}, Risk Level: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/canser", response_model=CancerPrediction)
async def predict_cancer(data: CancerInput):
    logger.info("Received cancer prediction request")
    
    if model_manager.canser_model is None or model_manager.canser_scaler is None:
        raise HTTPException(status_code=503, detail="Canser Model or Scaler not loaded")
    
    try:
        # Extract and order features
        features = np.array([[
            data.Age,
            data.Gender,
            data.Air_Pollution,
            data.Alcohol_use,
            data.Dust_Allergy,
            data.OccuPational_Hazards,
            data.Genetic_Risk,
            data.chronic_Lung_Disease,
            data.Balanced_Diet,
            data.Obesity,
            data.Smoking,
            data.Passive_Smoker,
            data.Chest_Pain,
            data.Coughing_of_Blood,
            data.Fatigue,
            data.Weight_Loss,
            data.Shortness_of_Breath,
            data.Wheezing,
            data.Swallowing_Difficulty,
            data.Clubbing_of_Finger_Nails,
            data.Frequent_Cold,
            data.Dry_Cough,
            data.Snoring
        ]])
        
        # Scale features if scaler is available
        features_scaled = model_manager.canser_scaler.transform(features) if model_manager.canser_scaler else features
        
        # Make prediction
        prediction = model_manager.canser_model.predict(features_scaled)[0]
        
        # Get probability scores if available
        if hasattr(model_manager.canser_model, "predict_proba"):
            proba = model_manager.canser_model.predict_proba(features_scaled)[0]
            probabilities = {f"Level_{i+1}": float(p) for i, p in enumerate(proba)}
        else:
            # For models without predict_proba
            probabilities = {f"Level_{prediction}": 1.0}
        
        # Get risk level description
        risk_level = get_canser_risk_level(prediction)
        
        # Perform detailed analysis of each parameter
        detailed_analyses = [
            analyze_canser_parameter('Age', data.Age),
            analyze_canser_parameter('Gender', data.Gender),
            analyze_canser_parameter('Air_Pollution', data.Air_Pollution),
            analyze_canser_parameter('Alcohol_use', data.Alcohol_use),
            analyze_canser_parameter('Dust_Allergy', data.Dust_Allergy),
            analyze_canser_parameter('OccuPational_Hazards', data.OccuPational_Hazards),
            analyze_canser_parameter('Genetic_Risk', data.Genetic_Risk),
            analyze_canser_parameter('chronic_Lung_Disease', data.chronic_Lung_Disease),
            analyze_canser_parameter('Balanced_Diet', data.Balanced_Diet),
            analyze_canser_parameter('Obesity', data.Obesity),
            analyze_canser_parameter('Smoking', data.Smoking),
            analyze_canser_parameter('Passive_Smoker', data.Passive_Smoker),
            analyze_canser_parameter('Chest_Pain', data.Chest_Pain),
            analyze_canser_parameter('Coughing_of_Blood', data.Coughing_of_Blood),
            analyze_canser_parameter('Fatigue', data.Fatigue),
            analyze_canser_parameter('Weight_Loss', data.Weight_Loss),
            analyze_canser_parameter('Shortness_of_Breath', data.Shortness_of_Breath),
            analyze_canser_parameter('Wheezing', data.Wheezing),
            analyze_canser_parameter('Swallowing_Difficulty', data.Swallowing_Difficulty),
            analyze_canser_parameter('Clubbing_of_Finger_Nails', data.Clubbing_of_Finger_Nails),
            analyze_canser_parameter('Frequent_Cold', data.Frequent_Cold),
            analyze_canser_parameter('Dry_Cough', data.Dry_Cough),
            analyze_canser_parameter('Snoring', data.Snoring)
        ]
        
        # Generate overall recommendation
        overall_recommendation = generate_canser_overall_recommendation(detailed_analyses, prediction)
        
        # Prepare response
        response = CancerPrediction(
            prediction=int(prediction),
            probability=probabilities,
            risk_level=risk_level,
            detailed_analysis=detailed_analyses,
            overall_recommendation=overall_recommendation,
            input_values=data.model_dump()
        )
        
        logger.info(f"Cancer prediction complete: Level {prediction}, Risk Level: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
