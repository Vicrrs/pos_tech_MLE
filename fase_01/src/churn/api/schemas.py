"""Contratos de entrada e saída da API."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

CUSTOMER_EXAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}


class CustomerFeatures(BaseModel):
    """Dados de um cliente, com os mesmos nomes de coluna do dataset de treino."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [CUSTOMER_EXAMPLE]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def check_service_consistency(self) -> Self:
        """Rejeita combinações de serviços que não existem nos dados de treino."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines é 'No phone service' somente se PhoneService é 'No'.")
        no_internet = self.InternetService == "No"
        for addon in INTERNET_ADDONS:
            if no_internet != (getattr(self, addon) == "No internet service"):
                raise ValueError(f"{addon} é 'No internet service' somente sem internet.")
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Propensão estimada de churn")
    churn_prediction: bool = Field(description="True se a propensão atinge o limiar de decisão")
    threshold: float = Field(description="Limiar de decisão usado")
    model_name: str

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool

    model_config = ConfigDict(protected_namespaces=())
