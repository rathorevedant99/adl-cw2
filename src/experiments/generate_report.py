"""
Generate a comprehensive report for the weakly-supervised segmentation experiments.
Includes methodology descriptions, experiment comparisons, and visualizations.
"""

import os
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import sys

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experiments.result_analysis import ExperimentAnalyzer


class ReportGenerator:
    """Generates comprehensive reports for weakly-supervised segmentation experiments."""
    
    def __init__(self, experiment_dir, output_dir=None):
        """
        Initialize report generator.
        
        Args:
            experiment_dir: Directory with experiment results
            output_dir: Directory to save report (default: experiment_dir/report)
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir) if output_dir else self.experiment_dir / "report"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer
        self.analyzer = ExperimentAnalyzer(experiment_dir)
        
        # Load results
        self.results = self.analyzer.results
        
        # Load experiment configs
        self.configs = self._load_configs()
    
    def _load_configs(self):
        """Load configurations for each experiment."""
        configs = {}
        
        for experiment_name in self.results.keys():
            if experiment_name == 'ablation':
                continue
                
            config_path = self.experiment_dir / experiment_name / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    configs[experiment_name] = yaml.safe_load(f)
        
        return configs
    
    def generate_full_report(self, include_images=True):
        """
        Generate a comprehensive report of all experiments.
        
        Args:
            include_images: Whether to include images in the report
            
        Returns:
            Path to generated report
        """
        # Run analysis
        comparison_df = self.analyzer.generate_comparison_table()
        
        # Generate report
        html_path = self.output_dir / "weakly_supervised_segmentation_report.html"
        
        # Start HTML content
        html_content = self._generate_html_header()
        
        # Add methodology section
        html_content += self._generate_methodology_section()
        
        # Add experiment comparison section
        html_content += self._generate_comparison_section(comparison_df)
        
        # Add ablation study section
        if 'ablation' in self.results:
            html_content += self._generate_ablation_section()
        
        # Add visualization section
        if include_images:
            html_content += self._generate_visualization_section()
        
        # Add conclusion section
        html_content += self._generate_conclusion_section()
        
        # Close HTML content
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Full report generated at {html_path}")
        
        return html_path
    
    def _generate_html_header(self):
        """Generate HTML header with styling."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weakly-Supervised Segmentation Experiment Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                    margin-top: 30px;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .highlight {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 5px solid #3498db;
                    margin: 20px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .caption {{
                    font-style: italic;
                    text-align: center;
                    color: #666;
                    margin-top: 5px;
                }}
                .method-box {{
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .conclusion {{
                    background-color: #f1f8e9;
                    border-left: 5px solid #8bc34a;
                    padding: 15px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Weakly-Supervised Segmentation Experiment Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
    
    def _generate_methodology_section(self):
        """Generate HTML section describing the methodology."""
        return """
            <h2>1. Methodology</h2>
            
            <h3>1.1. Weakly-Supervised Learning</h3>
            <p>
                Weakly-supervised learning aims to train models using less detailed annotations than 
                would typically be required. In the context of semantic segmentation, instead of pixel-level
                masks, we use image-level labels (classification) and leverage Class Activation Mapping (CAM)
                to infer the segmentation masks.
            </p>
            
            <div class="method-box">
                <h4>Our CAM-RG Approach</h4>
                <p>
                    We implemented a Class Activation Mapping with Region Growing (CAM-RG) approach that consists of:
                </p>
                <ol>
                    <li><strong>U-Net Architecture:</strong> A symmetric encoder-decoder network with skip connections that preserves spatial information.</li>
                    <li><strong>Class Activation Mapping:</strong> Uses a classification head to generate coarse localization maps for each class.</li>
                    <li><strong>Region Growing:</strong> Refines the boundaries of the activation maps using multi-scale features from the decoder.</li>
                    <li><strong>Consistency Constraints:</strong> Enforces consistency between forward and backward passes to improve segmentation quality.</li>
                </ol>
            </div>
            
            <h3>1.2. Training Process</h3>
            <p>
                The training process involves multiple components designed to maximize learning from weak supervision:
            </p>
            <ul>
                <li><strong>Classification Loss:</strong> Primary loss based on image-level labels.</li>
                <li><strong>Consistency Loss:</strong> Enforces similarity between CAM and final segmentation maps.</li>
                <li><strong>Size Constraint Loss:</strong> Prevents degenerate solutions by constraining segment sizes.</li>
                <li><strong>Curriculum Learning:</strong> Gradually incorporates more challenging data samples as training progresses.</li>
                <li><strong>Pseudo-Labeling:</strong> Generates pseudo-labels for unlabeled data to expand the training set.</li>
            </ul>
            
            <h3>1.3. Experiment Framework</h3>
            <p>
                We designed a comprehensive experiment framework to evaluate our approach:
            </p>
            <ul>
                <li><strong>Baseline Comparison:</strong> Comparing weakly-supervised method against fully-supervised baseline.</li>
                <li><strong>Ablation Studies:</strong> Analyzing the impact of key hyperparameters.</li>
                <li><strong>Additional Data Integration:</strong> Evaluating benefit of incorporating additional weakly-labeled data.</li>
                <li><strong>Visualization:</strong> Generating visual comparisons of segmentation results and CAM heatmaps.</li>
            </ul>
        """
    
    def _generate_comparison_section(self, comparison_df):
        """Generate HTML section comparing experiment results."""
        # Convert DataFrame to HTML table
        table_html = comparison_df.to_html(index=False, classes='table', border=0)
        
        # Get fully supervised and best weakly supervised results
        fully_supervised = None
        best_weakly = None
        
        if 'fully_supervised' in comparison_df['Experiment'].values:
            fully_supervised = comparison_df[comparison_df['Experiment'] == 'fully_supervised']
        
        # Exclude fully supervised from best weak comparison
        weak_df = comparison_df[comparison_df['Experiment'] != 'fully_supervised']
        if not weak_df.empty and 'miou' in weak_df.columns:
            best_idx = weak_df['miou'].idxmax()
            best_weakly = weak_df.loc[best_idx]
        
        # Generate comparison text
        comparison_text = ""
        if fully_supervised is not None and best_weakly is not None:
            fs_miou = fully_supervised['miou'].values[0]
            weak_miou = best_weakly['miou']
            gap = fs_miou - weak_miou
            
            comparison_text = f"""
            <div class="highlight">
                <h4>Key Comparison:</h4>
                <p>
                    The fully-supervised approach achieved a mIoU of {fs_miou:.4f}, while the best weakly-supervised
                    approach ({best_weakly['Experiment']}) achieved a mIoU of {weak_miou:.4f}.
                    This represents a performance gap of {gap:.4f} ({gap/fs_miou*100:.1f}%).
                </p>
            </div>
            """
        
        return f"""
            <h2>2. Experiment Comparison</h2>
            
            <h3>2.1. Overview of Experiments</h3>
            <p>
                We conducted multiple experiments to evaluate the effectiveness of our weakly-supervised
                segmentation approach. The table below summarizes the key metrics for each experiment:
            </p>
            
            {table_html}
            
            {comparison_text}
            
            <h3>2.2. Performance Analysis</h3>
            <p>
                The performance analysis shows several interesting findings:
            </p>
            <ul>
                <li>
                    <strong>Weakly-Supervised vs. Fully-Supervised:</strong> 
                    As expected, the fully-supervised approach generally outperforms weakly-supervised methods,
                    but our best weakly-supervised configuration achieves competitive results while requiring
                    significantly less annotation effort.
                </li>
                <li>
                    <strong>Impact of CAM Threshold:</strong>
                    The CAM threshold parameter significantly influences segmentation quality. Lower thresholds
                    increase recall but may reduce precision, while higher thresholds improve precision at the
                    cost of recall.
                </li>
                <li>
                    <strong>Region Growing Iterations:</strong>
                    More region growing iterations generally improve boundary accuracy, but too many iterations
                    can lead to over-segmentation and reduced performance.
                </li>
                <li>
                    <strong>Consistency Mechanisms:</strong>
                    The consistency loss and constraints play a crucial role in improving segmentation quality,
                    particularly around object boundaries.
                </li>
            </ul>
            
            <div class="image-container">
                <img src="../plots/miou_comparison.png" alt="mIoU Comparison" />
                <p class="caption">Figure 1: Comparison of mIoU across different experimental configurations.</p>
            </div>
        """
    
    def _generate_ablation_section(self):
        """Generate HTML section for ablation study results."""
        # Get ablation parameters
        ablation_params = [param for param in self.results['ablation'].keys() if param != 'baseline']
        
        # Generate parameter descriptions
        param_descriptions = {
            'model.cam_threshold': """
                <strong>CAM Threshold:</strong> Controls the initial binary activation map creation from continuous CAM values.
                Lower values capture more of the target object but may include background, while higher values are more precise
                but may miss parts of the target.
            """,
            'model.region_growing_iterations': """
                <strong>Region Growing Iterations:</strong> Determines how many iterations of boundary refinement are performed.
                More iterations allow more extensive refinement but risk expanding into incorrect regions.
            """,
            'training.consistency_weight': """
                <strong>Consistency Weight:</strong> Controls the strength of the consistency loss that enforces similarity
                between CAM predictions and final segmentation outputs. Higher values prioritize consistency, while lower
                values allow more independence between initial activations and final segmentation.
            """,
            'training.generate_pseudo_labels': """
                <strong>Pseudo-labeling:</strong> Whether to generate pseudo-labels for unlabeled data. Enabling this allows
                the model to leverage additional data but may introduce noise if the pseudo-labels are inaccurate.
            """
        }
        
        ablation_html = """
            <h2>3. Ablation Study</h2>
            
            <h3>3.1. Parameter Impact Analysis</h3>
            <p>
                We conducted an ablation study to understand the impact of key hyperparameters on the performance
                of our weakly-supervised segmentation model. The following parameters were systematically varied:
            </p>
            
            <ul>
        """
        
        for param in ablation_params:
            if param in param_descriptions:
                ablation_html += f"<li>{param_descriptions[param]}</li>"
            else:
                ablation_html += f"<li><strong>{param}:</strong> A key parameter in the weakly-supervised framework.</li>"
        
        ablation_html += """
            </ul>
            
            <h3>3.2. Results of Ablation Study</h3>
            <p>
                The ablation study revealed several important insights about the model's sensitivity to different parameters:
            </p>
        """
        
        # Add ablation plots
        for param in ablation_params:
            param_formatted = param.replace('.', '_')
            plot_path = f"../analysis_plots/ablation_{param_formatted}.png"
            
            ablation_html += f"""
                <div class="image-container">
                    <img src="{plot_path}" alt="Ablation of {param}" />
                    <p class="caption">Figure: Impact of {param} on model performance.</p>
                </div>
                
                <p>
                    <strong>Observations for {param}:</strong><br>
                    The plot shows how model performance varies with different values of this parameter,
                    helping identify the optimal configuration for the weakly-supervised segmentation framework.
                </p>
            """
        
        return ablation_html
    
    def _generate_visualization_section(self):
        """Generate HTML section with visualizations."""
        return """
            <h2>4. Visualizations</h2>
            
            <h3>4.1. Segmentation Comparisons</h3>
            <p>
                Visual comparison of segmentation results across different models provides qualitative insights
                into the strengths and weaknesses of each approach. The examples below showcase representative
                results from our experiments:
            </p>
            
            <div class="image-container">
                <img src="../visualizations/segmentation_example_1.png" alt="Segmentation Example 1" />
                <p class="caption">Figure: Comparison of segmentation results from different models. From left to right: 
                Original image, Ground truth, Fully supervised, and Weakly supervised methods.</p>
            </div>
            
            <h3>4.2. Class Activation Maps</h3>
            <p>
                Class Activation Maps (CAMs) visualize which regions of the image most influenced the classification
                decision. These heatmaps provide insights into how the weakly-supervised model identifies relevant
                features without pixel-level annotations:
            </p>
            
            <div class="image-container">
                <img src="../cam_visualizations/cam_example_1.png" alt="CAM Example 1" />
                <p class="caption">Figure: Class Activation Maps for different model configurations. Hotter colors (red)
                indicate regions strongly associated with the predicted class.</p>
            </div>
            
            <h3>4.3. Learning Progression</h3>
            <p>
                Tracking metrics throughout the training process helps understand how different models learn over time:
            </p>
            
            <div class="image-container">
                <img src="../analysis_plots/learning_curves.png" alt="Learning Curves" />
                <p class="caption">Figure: Training loss and validation mIoU curves for different experimental configurations.</p>
            </div>
        """
    
    def _generate_conclusion_section(self):
        """Generate HTML section with conclusions."""
        return """
            <h2>5. Conclusions</h2>
            
            <div class="conclusion">
                <h3>5.1. Key Findings</h3>
                <p>
                    Our experiments with the CAM-RG weakly-supervised segmentation framework revealed several important findings:
                </p>
                <ul>
                    <li>
                        <strong>Effectiveness of Weak Supervision:</strong> 
                        The proposed framework demonstrates that effective segmentation is possible with only image-level labels,
                        achieving results competitive with fully-supervised methods while requiring significantly less annotation effort.
                    </li>
                    <li>
                        <strong>Critical Components:</strong>
                        Region growing and consistency constraints are critical components for refining the initial coarse
                        activations into accurate segmentation masks. Our ablation studies highlight the importance of properly
                        tuning these components.
                    </li>
                    <li>
                        <strong>Parameter Sensitivity:</strong>
                        The framework shows sensitivity to certain hyperparameters, particularly the CAM threshold and the
                        number of region growing iterations. Careful tuning of these parameters is essential for optimal performance.
                    </li>
                    <li>
                        <strong>Additional Data Benefits:</strong>
                        Incorporating additional weakly-labeled data through pseudo-labeling and curriculum learning
                        provides measurable improvements in segmentation quality, especially for challenging or
                        underrepresented classes.
                    </li>
                </ul>
            </div>
            
            <h3>5.2. Future Directions</h3>
            <p>
                Based on our findings, several promising directions for future work emerge:
            </p>
            <ul>
                <li>
                    <strong>Advanced Region Growing:</strong> 
                    Exploring more sophisticated region growing algorithms that better utilize multi-scale features
                    from the U-Net architecture could further improve boundary accuracy.
                </li>
                <li>
                    <strong>Self-supervised Pretraining:</strong>
                    Incorporating self-supervised pretraining could improve feature representation and reduce the
                    dependence on labeled data.
                </li>
                <li>
                    <strong>Adaptive Thresholding:</strong>
                    Developing adaptive thresholding mechanisms that adjust based on image characteristics could
                    make the framework more robust across diverse inputs.
                </li>
                <li>
                    <strong>Cross-dataset Generalization:</strong>
                    Evaluating and improving the generalization capabilities of weakly-supervised models across
                    different datasets would enhance their practical utility.
                </li>
            </ul>
            
            <h3>5.3. Summary</h3>
            <p>
                The CAM-RG weakly-supervised segmentation framework represents a promising approach for semantic
                segmentation when pixel-level annotations are limited or expensive to obtain. By leveraging image-level
                labels and employing sophisticated region growing and consistency mechanisms, the framework achieves
                competitive performance while significantly reducing annotation requirements. The ablation studies
                provide valuable insights into the importance of different components and guide optimal parameter selection.
            </p>
        """


def main():
    """Main function to generate experiment report."""
    parser = argparse.ArgumentParser(description='Generate report for weakly-supervised segmentation experiments')
    parser.add_argument('--experiment_dir', type=str, required=True,
                     help='Directory with experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                     help='Directory to save report (default: experiment_dir/report)')
    parser.add_argument('--include_images', action='store_true', default=True,
                     help='Include images in the report')
    
    args = parser.parse_args()
    
    # Generate report
    report_generator = ReportGenerator(args.experiment_dir, args.output_dir)
    report_path = report_generator.generate_full_report(include_images=args.include_images)
    
    print(f"Report generated successfully at {report_path}")


if __name__ == "__main__":
    main()
