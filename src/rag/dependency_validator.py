"""
Dependency Validation and Management System for RAG
Provides utilities for checking dependencies, graceful degradation, and installation guidance.
"""

import importlib
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyStatus(Enum):
    """Status of dependency validation"""
    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    module_name: str
    required: bool
    version_required: Optional[str] = None
    status: DependencyStatus = DependencyStatus.MISSING
    installed_version: Optional[str] = None
    error_message: Optional[str] = None
    installation_command: Optional[str] = None

@dataclass
class DependencyValidationResult:
    """Result of dependency validation"""
    overall_status: str  # 'healthy', 'degraded', 'failed'
    critical_missing: List[str]
    optional_missing: List[str]
    available_dependencies: List[str]
    installation_instructions: Dict[str, str]
    degraded_features: List[str]
    error_details: Dict[str, str]

class DependencyValidator:
    """
    Validates and manages RAG system dependencies
    """
    
    def __init__(self):
        self.dependencies = self._define_dependencies()
        self.validation_cache = {}
    
    def _define_dependencies(self) -> Dict[str, DependencyInfo]:
        """Define all RAG system dependencies"""
        return {
            'sentence_transformers': DependencyInfo(
                name='Sentence Transformers',
                module_name='sentence_transformers',
                required=True,
                installation_command='pip install sentence-transformers',
            ),
            'faiss': DependencyInfo(
                name='FAISS',
                module_name='faiss',
                required=True,
                installation_command='pip install faiss-cpu',
            ),
            'numpy': DependencyInfo(
                name='NumPy',
                module_name='numpy',
                required=True,
                installation_command='pip install numpy',
            ),
            'pandas': DependencyInfo(
                name='Pandas',
                module_name='pandas',
                required=True,
                installation_command='pip install pandas',
            ),
            'sqlite3': DependencyInfo(
                name='SQLite3',
                module_name='sqlite3',
                required=True,
                installation_command='Built-in Python module',
            ),
            'pickle': DependencyInfo(
                name='Pickle',
                module_name='pickle',
                required=True,
                installation_command='Built-in Python module',
            ),
            'torch': DependencyInfo(
                name='PyTorch',
                module_name='torch',
                required=False,
                installation_command='pip install torch',
            ),
            'transformers': DependencyInfo(
                name='Transformers',
                module_name='transformers',
                required=False,
                installation_command='pip install transformers',
            ),
        }
    
    def check_dependency(self, dep_name: str) -> DependencyInfo:
        """
        Check a single dependency
        
        Args:
            dep_name: Name of the dependency to check
            
        Returns:
            DependencyInfo with validation results
        """
        if dep_name not in self.dependencies:
            raise ValueError(f"Unknown dependency: {dep_name}")
        
        dep_info = self.dependencies[dep_name]
        
        try:
            # Try to import the module
            module = importlib.import_module(dep_info.module_name)
            
            # Check if module has version attribute
            version = None
            for version_attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, version_attr):
                    version = getattr(module, version_attr)
                    break
            
            dep_info.status = DependencyStatus.AVAILABLE
            dep_info.installed_version = str(version) if version else "Unknown"
            dep_info.error_message = None
            
            logger.debug(f"Dependency {dep_name} is available (version: {version})")
            
        except ImportError as e:
            dep_info.status = DependencyStatus.MISSING
            dep_info.error_message = str(e)
            logger.warning(f"Dependency {dep_name} is missing: {e}")
            
        except Exception as e:
            dep_info.status = DependencyStatus.IMPORT_ERROR
            dep_info.error_message = str(e)
            logger.error(f"Error checking dependency {dep_name}: {e}")
        
        return dep_info
    
    def check_critical_dependencies(self) -> DependencyValidationResult:
        """
        Check all critical dependencies required for RAG system
        
        Returns:
            DependencyValidationResult with validation status
        """
        critical_deps = [name for name, info in self.dependencies.items() if info.required]
        return self._validate_dependencies(critical_deps, check_optional=False)
    
    def check_optional_dependencies(self) -> DependencyValidationResult:
        """
        Check optional dependencies for enhanced features
        
        Returns:
            DependencyValidationResult with validation status
        """
        optional_deps = [name for name, info in self.dependencies.items() if not info.required]
        return self._validate_dependencies(optional_deps, check_optional=True)
    
    def check_all_dependencies(self) -> DependencyValidationResult:
        """
        Check all dependencies (critical and optional)
        
        Returns:
            DependencyValidationResult with complete validation status
        """
        all_deps = list(self.dependencies.keys())
        return self._validate_dependencies(all_deps, check_optional=True)
    
    def _validate_dependencies(self, dep_names: List[str], check_optional: bool = True) -> DependencyValidationResult:
        """
        Validate a list of dependencies
        
        Args:
            dep_names: List of dependency names to check
            check_optional: Whether to include optional dependencies in the check
            
        Returns:
            DependencyValidationResult with validation results
        """
        critical_missing = []
        optional_missing = []
        available_dependencies = []
        installation_instructions = {}
        degraded_features = []
        error_details = {}
        
        for dep_name in dep_names:
            dep_info = self.check_dependency(dep_name)
            
            if dep_info.status == DependencyStatus.AVAILABLE:
                available_dependencies.append(dep_name)
            else:
                if dep_info.required:
                    critical_missing.append(dep_name)
                else:
                    optional_missing.append(dep_name)
                
                # Add installation instructions
                if dep_info.installation_command:
                    installation_instructions[dep_name] = dep_info.installation_command
                
                # Add error details
                if dep_info.error_message:
                    error_details[dep_name] = dep_info.error_message
                
                # Determine degraded features
                degraded_features.extend(self._get_degraded_features(dep_name))
        
        # Determine overall status
        if critical_missing:
            overall_status = "failed"
        elif optional_missing:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return DependencyValidationResult(
            overall_status=overall_status,
            critical_missing=critical_missing,
            optional_missing=optional_missing,
            available_dependencies=available_dependencies,
            installation_instructions=installation_instructions,
            degraded_features=degraded_features,
            error_details=error_details
        )
    
    def _get_degraded_features(self, dep_name: str) -> List[str]:
        """
        Get list of features that will be degraded if dependency is missing
        
        Args:
            dep_name: Name of the missing dependency
            
        Returns:
            List of degraded feature descriptions
        """
        feature_map = {
            'sentence_transformers': [
                'Vector embeddings generation',
                'Semantic similarity search',
                'Document retrieval functionality'
            ],
            'faiss': [
                'Fast similarity search',
                'Vector index optimization',
                'Large-scale document retrieval'
            ],
            'torch': [
                'Advanced neural network models',
                'GPU acceleration for embeddings',
                'Custom transformer models'
            ],
            'transformers': [
                'Advanced language models',
                'Custom tokenization',
                'Model fine-tuning capabilities'
            ]
        }
        
        return feature_map.get(dep_name, [f"Features requiring {dep_name}"])
    
    def validate_sentence_transformers(self) -> bool:
        """
        Specifically validate sentence_transformers availability
        
        Returns:
            Boolean indicating if sentence_transformers is available
        """
        dep_info = self.check_dependency('sentence_transformers')
        return dep_info.status == DependencyStatus.AVAILABLE
    
    def get_installation_instructions(self, missing_deps: List[str]) -> str:
        """
        Generate installation instructions for missing dependencies
        
        Args:
            missing_deps: List of missing dependency names
            
        Returns:
            Formatted installation instructions
        """
        if not missing_deps:
            return "All dependencies are installed."
        
        instructions = ["To install missing dependencies, run the following commands:\n"]
        
        for dep_name in missing_deps:
            if dep_name in self.dependencies:
                dep_info = self.dependencies[dep_name]
                if dep_info.installation_command:
                    instructions.append(f"# Install {dep_info.name}")
                    instructions.append(dep_info.installation_command)
                    instructions.append("")
        
        # Add additional notes
        instructions.extend([
            "Additional Notes:",
            "- For GPU support with FAISS, use: pip install faiss-gpu",
            "- For PyTorch with GPU support, visit: https://pytorch.org/get-started/locally/",
            "- If you encounter permission issues, try using --user flag: pip install --user <package>",
            "- For conda environments, use: conda install -c conda-forge <package>"
        ])
        
        return "\n".join(instructions)
    
    def get_graceful_degradation_message(self, missing_deps: List[str]) -> str:
        """
        Generate user-friendly message about degraded functionality
        
        Args:
            missing_deps: List of missing dependency names
            
        Returns:
            User-friendly degradation message
        """
        if not missing_deps:
            return "All RAG system features are available."
        
        messages = ["⚠️ Some RAG system features are unavailable due to missing dependencies:\n"]
        
        for dep_name in missing_deps:
            if dep_name in self.dependencies:
                dep_info = self.dependencies[dep_name]
                degraded_features = self._get_degraded_features(dep_name)
                
                messages.append(f"Missing {dep_info.name}:")
                for feature in degraded_features:
                    messages.append(f"  • {feature}")
                messages.append("")
        
        messages.append("To restore full functionality, please install the missing dependencies.")
        messages.append("Use the installation instructions provided by the system administrator.")
        
        return "\n".join(messages)
    
    def create_fallback_model(self):
        """
        Create a fallback model when sentence_transformers is not available
        
        Returns:
            Fallback model object or None
        """
        try:
            # Try to use a simple TF-IDF based approach as fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            logger.info("Creating TF-IDF fallback model for sentence embeddings")
            
            class TfidfFallbackModel:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    self.fitted = False
                
                def encode(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    if not self.fitted:
                        # Fit on the provided texts (simple approach)
                        self.vectorizer.fit(texts)
                        self.fitted = True
                    
                    return self.vectorizer.transform(texts).toarray()
            
            return TfidfFallbackModel()
            
        except ImportError:
            logger.warning("sklearn not available for fallback model")
            return None
    
    def get_system_status_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system status report
        
        Returns:
            Dictionary with complete dependency status
        """
        validation_result = self.check_all_dependencies()
        
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": validation_result.overall_status,
            "summary": {
                "total_dependencies": len(self.dependencies),
                "available": len(validation_result.available_dependencies),
                "critical_missing": len(validation_result.critical_missing),
                "optional_missing": len(validation_result.optional_missing)
            },
            "critical_dependencies": {
                "available": [dep for dep in validation_result.available_dependencies 
                            if self.dependencies[dep].required],
                "missing": validation_result.critical_missing
            },
            "optional_dependencies": {
                "available": [dep for dep in validation_result.available_dependencies 
                            if not self.dependencies[dep].required],
                "missing": validation_result.optional_missing
            },
            "degraded_features": validation_result.degraded_features,
            "installation_instructions": validation_result.installation_instructions,
            "error_details": validation_result.error_details,
            "recommendations": self._generate_recommendations(validation_result)
        }
    
    def _generate_recommendations(self, validation_result: DependencyValidationResult) -> List[str]:
        """
        Generate actionable recommendations based on validation results
        
        Args:
            validation_result: Validation results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if validation_result.critical_missing:
            recommendations.append(
                f"CRITICAL: Install missing dependencies: {', '.join(validation_result.critical_missing)}"
            )
            recommendations.append("RAG system cannot function without critical dependencies")
        
        if validation_result.optional_missing:
            recommendations.append(
                f"OPTIONAL: Consider installing for enhanced features: {', '.join(validation_result.optional_missing)}"
            )
        
        if validation_result.overall_status == "healthy":
            recommendations.append("All dependencies are available - RAG system fully functional")
        
        if 'sentence_transformers' in validation_result.critical_missing:
            recommendations.append("Priority: Install sentence-transformers for core RAG functionality")
        
        if 'faiss' in validation_result.critical_missing:
            recommendations.append("Priority: Install FAISS for efficient vector search")
        
        return recommendations

# Global instance
dependency_validator = DependencyValidator()