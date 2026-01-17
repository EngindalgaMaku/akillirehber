"""LLM Models API endpoints for managing custom LLM models."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.db_models import User, CustomLLMModel
from app.models.schemas import (
    CustomLLMModelCreate,
    CustomLLMModelResponse,
    CustomLLMModelListResponse,
    LLMModelsResponse,
)
from app.services.auth_service import get_current_teacher
from app.services.llm_providers import LLM_PROVIDERS

router = APIRouter(prefix="/api/llm-models", tags=["llm-models"])


@router.get("", response_model=CustomLLMModelListResponse)
async def get_custom_llm_models(
    provider: str = None,
    db: Session = Depends(get_db),
):
    """
    Get all custom LLM models.
    
    Optionally filter by provider.
    """
    query = db.query(CustomLLMModel).filter(CustomLLMModel.is_active == True)
    
    if provider:
        query = query.filter(CustomLLMModel.provider == provider)
    
    models = query.order_by(CustomLLMModel.created_at.desc()).all()
    
    return CustomLLMModelListResponse(
        models=[CustomLLMModelResponse.model_validate(m) for m in models],
        total=len(models)
    )


@router.get("/by-provider/{provider}", response_model=LLMModelsResponse)
async def get_models_by_provider(
    provider: str,
    db: Session = Depends(get_db),
):
    """
    Get all models (default + custom) for a specific provider.
    """
    # Get default models
    default_models = []
    if provider in LLM_PROVIDERS:
        default_models = LLM_PROVIDERS[provider]["models"]
    
    # Get custom models
    custom_models = db.query(CustomLLMModel).filter(
        CustomLLMModel.provider == provider,
        CustomLLMModel.is_active == True
    ).order_by(CustomLLMModel.created_at.desc()).all()
    
    return LLMModelsResponse(
        default_models=default_models,
        custom_models=[
            CustomLLMModelResponse.model_validate(m) for m in custom_models
        ]
    )


@router.post("", response_model=CustomLLMModelResponse, status_code=201)
async def create_custom_llm_model(
    request: CustomLLMModelCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Create a new custom LLM model.
    
    Only teachers can add custom models.
    """
    # Validate provider exists
    if request.provider not in LLM_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider: {request.provider}. "
                   f"Available: {list(LLM_PROVIDERS.keys())}"
        )
    
    # Check if model already exists (either in defaults or custom)
    default_models = LLM_PROVIDERS[request.provider]["models"]
    if request.model_id in default_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_id}' already exists as default"
        )
    
    existing = db.query(CustomLLMModel).filter(
        CustomLLMModel.provider == request.provider,
        CustomLLMModel.model_id == request.model_id,
        CustomLLMModel.is_active == True
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_id}' already exists for this provider"
        )
    
    # Create new model
    model = CustomLLMModel(
        provider=request.provider,
        model_id=request.model_id,
        display_name=request.display_name,
        created_by=current_user.id
    )
    
    db.add(model)
    db.commit()
    db.refresh(model)
    
    return CustomLLMModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
async def delete_custom_llm_model(
    model_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete a custom LLM model.
    
    Only teachers can delete custom models.
    """
    model = db.query(CustomLLMModel).filter(
        CustomLLMModel.id == model_id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    # Soft delete by setting is_active to False
    model.is_active = False
    db.commit()
    
    return None


@router.get("/providers")
async def get_available_providers():
    """Get list of available LLM providers."""
    return {
        "providers": list(LLM_PROVIDERS.keys())
    }