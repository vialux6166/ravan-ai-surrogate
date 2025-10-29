"""
Model Interpretability for Ravan Quantum-ML System
Uses SHAP (SHapley Additive exPlanations) for feature importance analysis
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Model interpretability using SHAP values
    
    Provides insights into which input parameters most influence
    each quantum observable prediction.
    """
    
    def __init__(self, model, model_type: str = 'mlp'):
        """
        Initialize model interpreter
        
        Args:
            model: Trained ML model (MLPRegressor or XGBoostRegressor)
            model_type: Type of model ('mlp' or 'xgboost')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
        logger.info(f"ModelInterpreter initialized for {model_type} model")
    
    def explain_predictions(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        parameter_names: List[str],
        observable_names: List[str],
        n_background: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for test samples
        
        Args:
            X_background: Background dataset for SHAP (training data sample)
            X_test: Test samples to explain
            parameter_names: Names of input parameters
            observable_names: Names of output observables
            n_background: Number of background samples to use
            
        Returns:
            SHAP values array (n_test, n_features, n_outputs)
        """
        logger.info(f"Computing SHAP values for {len(X_test)} test samples...")
        
        try:
            import shap
        except ImportError:
            logger.error("SHAP library not installed. Install with: pip install shap")
            raise
        
        # Sample background data if too large
        if len(X_background) > n_background:
            indices = np.random.choice(len(X_background), n_background, replace=False)
            X_background = X_background[indices]
        
        logger.info(f"Using {len(X_background)} background samples")
        
        # Create appropriate explainer based on model type
        if self.model_type == 'mlp':
            # For neural networks, use DeepExplainer or KernelExplainer
            logger.info("Creating SHAP explainer for MLP (using KernelExplainer)...")
            
            # Wrap model prediction for SHAP
            def model_predict(X):
                return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(model_predict, X_background)
            
        elif self.model_type == 'xgboost':
            # For tree-based models, use TreeExplainer
            logger.info("Creating SHAP explainer for XGBoost (using TreeExplainer)...")
            self.explainer = shap.TreeExplainer(self.model.model)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Compute SHAP values
        logger.info("Computing SHAP values (this may take a few minutes)...")
        self.shap_values = self.explainer.shap_values(X_test)
        
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # Multi-output: list of arrays
            self.shap_values = np.array(self.shap_values)
            # Transpose to (n_samples, n_features, n_outputs)
            self.shap_values = np.transpose(self.shap_values, (1, 2, 0))
        
        logger.info(f"SHAP values computed: shape={self.shap_values.shape}")
        
        return self.shap_values
    
    def plot_summary(
        self,
        X_test: np.ndarray,
        parameter_names: List[str],
        observable_names: List[str],
        output_dir: str = './results/interpretability'
    ):
        """
        Create SHAP summary plots for feature importance
        
        Args:
            X_test: Test samples
            parameter_names: Names of input parameters
            observable_names: Names of output observables
            output_dir: Directory to save plots
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_predictions() first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shap
        
        logger.info("Generating SHAP summary plots...")
        
        # Plot for each observable
        n_observables = len(observable_names)
        
        for i, obs_name in enumerate(observable_names):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get SHAP values for this observable
            if self.shap_values.ndim == 3:
                shap_vals = self.shap_values[:, :, i]
            else:
                shap_vals = self.shap_values
            
            # Create summary plot
            shap.summary_plot(
                shap_vals,
                X_test,
                feature_names=parameter_names,
                show=False,
                plot_type='bar'
            )
            
            plt.title(f'Feature Importance for {obs_name}')
            plt.tight_layout()
            
            plot_path = output_dir / f'shap_summary_{obs_name}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
            plt.close()
        
        # Overall summary plot (all observables)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if self.shap_values.ndim == 3:
            # Average absolute SHAP values across all observables
            mean_shap = np.mean(np.abs(self.shap_values), axis=2)
        else:
            mean_shap = np.abs(self.shap_values)
        
        shap.summary_plot(
            mean_shap,
            X_test,
            feature_names=parameter_names,
            show=False,
            plot_type='bar'
        )
        
        plt.title('Overall Feature Importance (All Observables)')
        plt.tight_layout()
        
        plot_path = output_dir / 'shap_summary_overall.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {plot_path}")
        plt.close()
    
    def plot_force_plots(
        self,
        X_test: np.ndarray,
        parameter_names: List[str],
        observable_names: List[str],
        sample_indices: List[int] = [0, 1, 2],
        output_dir: str = './results/interpretability'
    ):
        """
        Generate force plots for individual predictions
        
        Args:
            X_test: Test samples
            parameter_names: Names of input parameters
            observable_names: Names of output observables
            sample_indices: Indices of samples to plot
            output_dir: Directory to save plots
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_predictions() first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shap
        
        logger.info(f"Generating force plots for {len(sample_indices)} samples...")
        
        for sample_idx in sample_indices:
            for obs_idx, obs_name in enumerate(observable_names):
                # Get SHAP values for this sample and observable
                if self.shap_values.ndim == 3:
                    shap_vals = self.shap_values[sample_idx, :, obs_idx]
                else:
                    shap_vals = self.shap_values[sample_idx, :]
                
                # Create force plot
                fig, ax = plt.subplots(figsize=(14, 3))
                
                # Manual force plot (SHAP's force plot is interactive HTML)
                # We'll create a bar plot showing contribution of each feature
                feature_contributions = shap_vals
                
                colors = ['red' if x < 0 else 'green' for x in feature_contributions]
                
                ax.barh(parameter_names, feature_contributions, color=colors, alpha=0.7)
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title(f'Feature Contributions for Sample {sample_idx} - {obs_name}')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                
                plot_path = output_dir / f'force_plot_sample{sample_idx}_{obs_name}.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Force plots saved to {output_dir}")
    
    def get_feature_importance(
        self,
        observable_names: List[str],
        parameter_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get feature importance rankings for each observable
        
        Args:
            observable_names: Names of output observables
            parameter_names: Names of input parameters
            
        Returns:
            Dictionary mapping observable names to feature importance arrays
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_predictions() first")
        
        importance_dict = {}
        
        for i, obs_name in enumerate(observable_names):
            if self.shap_values.ndim == 3:
                shap_vals = self.shap_values[:, :, i]
            else:
                shap_vals = self.shap_values
            
            # Mean absolute SHAP value for each feature
            importance = np.mean(np.abs(shap_vals), axis=0)
            importance_dict[obs_name] = importance
        
        return importance_dict
    
    def print_feature_rankings(
        self,
        observable_names: List[str],
        parameter_names: List[str],
        top_k: int = 5
    ):
        """
        Print top-k most important features for each observable
        
        Args:
            observable_names: Names of output observables
            parameter_names: Names of input parameters
            top_k: Number of top features to show
        """
        importance_dict = self.get_feature_importance(observable_names, parameter_names)
        
        print("\n" + "=" * 80)
        print("Feature Importance Rankings")
        print("=" * 80)
        
        for obs_name, importance in importance_dict.items():
            print(f"\n{obs_name}:")
            
            # Sort by importance
            sorted_indices = np.argsort(importance)[::-1]
            
            for rank, idx in enumerate(sorted_indices[:top_k], 1):
                param_name = parameter_names[idx]
                importance_val = importance[idx]
                print(f"  {rank}. {param_name}: {importance_val:.6f}")
        
        print("\n" + "=" * 80)


def analyze_model_interpretability(
    model,
    model_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    parameter_names: List[str],
    observable_names: List[str],
    output_dir: str = './results/interpretability'
) -> ModelInterpreter:
    """
    Complete interpretability analysis workflow
    
    Args:
        model: Trained model
        model_type: 'mlp' or 'xgboost'
        X_train: Training data (for background)
        X_test: Test data to explain
        parameter_names: Parameter names
        observable_names: Observable names
        output_dir: Output directory
        
    Returns:
        ModelInterpreter instance with computed SHAP values
    """
    logger.info("=" * 80)
    logger.info("Starting Model Interpretability Analysis")
    logger.info("=" * 80)
    
    # Create interpreter
    interpreter = ModelInterpreter(model, model_type)
    
    # Compute SHAP values
    interpreter.explain_predictions(
        X_background=X_train,
        X_test=X_test,
        parameter_names=parameter_names,
        observable_names=observable_names
    )
    
    # Generate visualizations
    interpreter.plot_summary(X_test, parameter_names, observable_names, output_dir)
    interpreter.plot_force_plots(X_test, parameter_names, observable_names, output_dir=output_dir)
    
    # Print rankings
    interpreter.print_feature_rankings(observable_names, parameter_names)
    
    logger.info("=" * 80)
    logger.info("Interpretability Analysis Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return interpreter


if __name__ == '__main__':
    print("Model Interpretability module loaded successfully!")
    print("Use analyze_model_interpretability() for complete analysis")
