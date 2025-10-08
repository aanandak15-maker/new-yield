"""
Cotton Data Extraction for India Agricultural Intelligence Platform
Extracts and processes cotton yield data from APY datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CottonDataExtractor:
    """Extract and process cotton data from agricultural datasets"""

    def __init__(self):
        self.main_data_dir = Path("cotton_data")
        self.main_data_dir.mkdir(exist_ok=True)

        # Cotton producing states
        self.cotton_states = [
            'PUNJAB', 'HARYANA', 'RAJASTHAN', 'UTTAR PRADESH',
            'MAHARASHTRA', 'GUJARAT', 'ANDHRA PRADESH', 'KARNATKA',
            'MADHYA PRADESH', 'TAMIL NADU'
        ]

    def main(self):
        """Main execution function"""
        print("🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM")
        print("👔 Cotton Data Extraction for Multi-State Model Training")
        print("=" * 60)

        print("👔 COTTON DATA EXTRACTION - PHASE 1")
        print("=" * 60)

        # Extract cotton data from APY dataset
        cotton_dataset = self.extract_cotton_data_from_apy()

        if cotton_dataset is not None:
            print(f"\n📊 Cotton Extraction Summary:")
            print(f"   • Total Cotton Records: {len(cotton_dataset)}")
            print(f"   • States Covered: {len(cotton_dataset['State_clean'].unique())}")
            print(f"   • Date Range: {cotton_dataset['Crop_Year'].min()} - {cotton_dataset['Crop_Year'].max()}")
            print(f"Average Yield: {cotton_dataset['yield_quintal_ha'].mean():.3f} q/ha")
            print(f"Median Yield: {cotton_dataset['yield_quintal_ha'].median():.3f} q/ha")
            print(f"Max Yield: {cotton_dataset['yield_quintal_ha'].max():.1f} q/ha")
            # Split by state and save
            self.split_and_save_cotton_data(cotton_dataset)

            # Create analysis visualizations
            self.create_cotton_analysis_plots(cotton_dataset)

            # Create summary report
            self.create_cotton_summary_report(cotton_dataset)

            print("\n✅ Cotton data extraction completed successfully!")
            return cotton_dataset

        return None

    def extract_cotton_data_from_apy(self):
        """Extract cotton data from APY dataset"""
        try:
            print("Loading APY.csv...")

            # Try to extract from zip file
            if Path("APY.csv.zip").exists():
                with zipfile.ZipFile("APY.csv.zip", 'r') as zip_ref:
                    # Find the CSV file in the zip
                    csv_files = [f for f in zip_ref.filelist if f.filename.endswith('.csv')]
                    if csv_files:
                        csv_file = csv_files[0].filename
                        print(f"Extracting {csv_file} from zip...")
                        with zip_ref.open(csv_file) as f:
                            df = pd.read_csv(f)
                    else:
                        print("No CSV file found in zip")
                        return None
            else:
                # Try direct CSV file
                csv_path = Path("APY.csv")
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                else:
                    print("❌ No APY dataset found (APY.csv or APY.csv.zip)")
                    return None

            print(f"APY Dataset loaded: {len(df)} records")
            print(f"Columns: {list(df.columns)}")

            # Filter for cotton
            cotton_df = df[df['Crop'].str.upper().str.contains('COTTON', na=False)].copy()
            print(f"Cotton records found: {len(cotton_df)}")

            if len(cotton_df) == 0:
                print("⚠️  No cotton records found in dataset")
                return None

            # Clean and standardize data
            cotton_cleaned = self._preprocess_cotton_data(cotton_df)

            # Filter for major cotton states
            major_cotton_df = cotton_cleaned[cotton_cleaned['State_clean'].isin(self.cotton_states)].copy()
            print(f"Filtered to major cotton states: {len(major_cotton_df)} records")

            if len(major_cotton_df) == 0:
                print("⚠️  No records found for major cotton states")
                return None

            return major_cotton_df

        except Exception as e:
            print(f"❌ Error extracting cotton data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _preprocess_cotton_data(self, cotton_df):
        """Preprocess cotton data for analysis"""

        print("🔧 Preprocessing cotton data...")

        # Handle column name variations
        column_mapping = {
            'State': 'State_clean',
            'District ': 'District_clean',
            'Crop_Year': 'Crop_Year',
            'Season': 'Season_clean',
            'Area ': 'Area',
            'Production': 'Production',
            'Yield': 'yield_quintal_ha'
        }

        # Rename columns if they exist
        existing_cols = {old: new for old, new in column_mapping.items() if old in cotton_df.columns}
        cotton_df = cotton_df.rename(columns=existing_cols)

        # Standardize state names
        state_mapping = {
            'Punjab': 'PUNJAB',
            'Haryana': 'HARYANA',
            'Rajasthan': 'RAJASTHAN',
            'Uttar Pradesh': 'UTTAR PRADESH',
            'Maharashtra': 'MAHARASHTRA',
            'Gujarat': 'GUJARAT',
            'Andhra Pradesh': 'ANDHRA PRADESH',
            'Karnataka': 'KARNATKA',
            'Madhya Pradesh': 'MADHYA PRADESH',
            'Tamil Nadu': 'TAMIL NADU'
        }

        cotton_df['State_clean'] = cotton_df['State_clean'].replace(state_mapping)

        # Clean season data
        season_mapping = {
            'Kharif': 'KHARIF',
            'Rabi': 'RABI',
            'Autumn': 'KHARIF',
            'Winter': 'RABI',
            'Summer': 'SUMMER'
        }
        if 'Season_clean' in cotton_df.columns:
            cotton_df['Season_clean'] = cotton_df['Season_clean'].replace(season_mapping)

        # Clean district names
        if 'District_clean' in cotton_df.columns:
            cotton_df['District_clean'] = cotton_df['District_clean'].str.strip()

        # Handle missing values
        cotton_df['Area'] = pd.to_numeric(cotton_df['Area'], errors='coerce')
        cotton_df['Production'] = pd.to_numeric(cotton_df['Production'], errors='coerce')
        cotton_df['yield_quintal_ha'] = pd.to_numeric(cotton_df['yield_quintal_ha'], errors='coerce')

        # Calculate yield where missing
        mask = cotton_df['yield_quintal_ha'].isna() & cotton_df['Area'].notna() & cotton_df['Production'].notna() & (cotton_df['Area'] > 0)
        cotton_df.loc[mask, 'yield_quintal_ha'] = cotton_df.loc[mask, 'Production'] / cotton_df.loc[mask, 'Area']

        # Drop rows with no yield data
        cotton_df = cotton_df.dropna(subset=['yield_quintal_ha', 'Area'])

        # Remove unrealistic yields (cotton typically 1-30 q/ha)
        cotton_df = cotton_df[(cotton_df['yield_quintal_ha'] >= 1) & (cotton_df['yield_quintal_ha'] <= 50)]

        print(f"Input columns: {list(cotton_df.columns)}")
        print(f"Preprocessed cotton data: {len(cotton_df)} records")

        return cotton_df

    def split_and_save_cotton_data(self, cotton_df):
        """Split cotton data by state and save"""

        print("💾 Saving cotton datasets...")

        # Save complete dataset
        complete_file = self.main_data_dir / "cotton_dataset_complete.csv"
        cotton_df.to_csv(complete_file, index=False)
        print(f"Saved complete cotton dataset: {len(cotton_df)} records")

        # Split by state
        for state in self.cotton_states:
            state_data = cotton_df[cotton_df['State_clean'] == state].copy()
            if len(state_data) > 0:
                state_file = self.main_data_dir / f"cotton_{state.lower()}.csv"
                state_data.to_csv(state_file, index=False)
                print(f"Saved {state.lower()} cotton data: {len(state_data)} records")

        # Create training and validation splits (temporal)
        if 'Crop_Year' in cotton_df.columns:
            training_data = cotton_df[cotton_df['Crop_Year'] <= 2018].copy()
            validation_data = cotton_df[cotton_df['Crop_Year'] >= 2019].copy()

            training_file = self.main_data_dir / "cotton_training_data.csv"
            validation_file = self.main_data_dir / "cotton_validation_data.csv"

            training_data.to_csv(training_file, index=False)
            validation_data.to_csv(validation_file, index=False)

            print(f"Saved training data: {len(training_data)} records (≤2018)")
            print(f"Saved validation data: {len(validation_data)} records (≥2019)")

    def create_cotton_analysis_plots(self, cotton_df):
        """Create analysis plots for cotton data"""

        print("📈 Creating cotton data visualizations...")

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Yield distribution by state
            state_yields = cotton_df.groupby('State_clean')['yield_quintal_ha'].describe()
            state_yields['mean'].plot(kind='bar', ax=axes[0,0], color='lightblue')
            axes[0,0].set_title('Average Cotton Yield by State (q/ha)')
            axes[0,0].set_ylabel('Yield (q/ha)')
            axes[0,0].tick_params(axis='x', rotation=45)

            # Yield trend over years
            if 'Crop_Year' in cotton_df.columns:
                yearly_trend = cotton_df.groupby('Crop_Year')['yield_quintal_ha'].mean()
                yearly_trend.plot(ax=axes[0,1], marker='o')
                axes[0,1].set_title('Cotton Yield Trend Over Years')
                axes[0,1].set_ylabel('Average Yield (q/ha)')
                axes[0,1].grid(True, alpha=0.3)

            # Yield distribution histogram
            axes[1,0].hist(cotton_df['yield_quintal_ha'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1,0].set_title('Cotton Yield Distribution')
            axes[1,0].set_xlabel('Yield (q/ha)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].axvline(cotton_df['yield_quintal_ha'].mean(), color='red', linestyle='--', label='.1f')
            axes[1,0].legend()

            # Area vs Yield scatter
            axes[1,1].scatter(cotton_df['Area'], cotton_df['yield_quintal_ha'], alpha=0.6, s=30, color='orange')
            axes[1,1].set_title('Cotton Area vs Yield Relationship')
            axes[1,1].set_xlabel('Area (hectares)')
            axes[1,1].set_ylabel('Yield (q/ha)')
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plots
            plots_file = self.main_data_dir / "cotton_data_analysis.png"
            plt.savefig(plots_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Visualizations saved to {plots_file}")

        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")

    def create_cotton_summary_report(self, cotton_df):
        """Create comprehensive cotton data summary"""

        summary = {
            "extraction_metadata": {
                "extraction_date": datetime.now().isoformat(),
                "source_dataset": "APY.csv",
                "extraction_method": "cotton_crop_filtering"
            },
            "dataset_summary": {
                "total_records": len(cotton_df),
                "unique_states": len(cotton_df['State_clean'].unique()) if 'State_clean' in cotton_df.columns else 0,
                "date_range": f"{cotton_df['Crop_Year'].min()}-{cotton_df['Crop_Year'].max()}" if 'Crop_Year' in cotton_df.columns else "unknown",
                "avg_yield": round(cotton_df['yield_quintal_ha'].mean(), 2),
                "median_yield": round(cotton_df['yield_quintal_ha'].median(), 2),
                "yield_std": round(cotton_df['yield_quintal_ha'].std(), 2)
            }
        }

        # State-wise summary
        if 'State_clean' in cotton_df.columns:
            state_summary = {}
            for state in cotton_df['State_clean'].unique():
                state_data = cotton_df[cotton_df['State_clean'] == state]
                state_summary[state] = {
                    "record_count": len(state_data),
                    "avg_yield": round(state_data['yield_quintal_ha'].mean(), 2),
                    "max_yield": round(state_data['yield_quintal_ha'].max(), 2),
                    "min_yield": round(state_data['yield_quintal_ha'].min(), 2)
                }

            top_states = sorted(state_summary.items(), key=lambda x: x[1]['record_count'], reverse=True)[:5]
            summary["top_states"] = {state: count for state, stats in top_states for count in [stats['record_count']]}

        # Save summary
        summary_file = self.main_data_dir / "cotton_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📋 Cotton summary saved: {summary_file}")

        return summary

def main():
    """Main execution"""
    extractor = CottonDataExtractor()
    return extractor.main()

if __name__ == "__main__":
    main()
