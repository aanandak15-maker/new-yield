"""
Maize Data Extraction for India Agricultural Intelligence Platform
Extracts and processes maize/corn yield data from APY datasets
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

class MaizeDataExtractor:
    """Extract and process maize data from agricultural datasets"""

    def __init__(self):
        self.main_data_dir = Path("maize_data")
        self.main_data_dir.mkdir(exist_ok=True)

        # Maize producing states
        self.maize_states = [
            'KARNATAKA', 'MAHARASHTRA', 'ANDHRA_PRADESH', 'BIHAR',
            'UTTAR_PRADESH', 'MADHYA_PRADESH', 'TAMIL_NADU',
            'RAJASTHAN', 'GUJARAT'
        ]

    def main(self):
        """Main execution function"""
        print("🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM")
        print("🌽 Maize Data Extraction for Multi-State Model Training")
        print("=" * 60)

        print("🌽 MAIZE DATA EXTRACTION - PHASE 1")
        print("=" * 60)

        # Extract maize data from APY dataset
        maize_dataset = self.extract_maize_data_from_apy()

        if maize_dataset is not None:
            print(f"\n📊 Maize Extraction Summary:")
            print(f"   • Total Maize Records: {len(maize_dataset)}")
            print(f"   • States Covered: {len(maize_dataset['State_clean'].unique())}")
            print(f"   • Date Range: {maize_dataset['Crop_Year'].min()} - {maize_dataset['Crop_Year'].max()}")
            print(f"Average Yield: {maize_dataset['yield_quintal_ha'].mean():.3f} q/ha")
            print(f"Median Yield: {maize_dataset['yield_quintal_ha'].median():.3f} q/ha")
            print(f"Max Yield: {maize_dataset['yield_quintal_ha'].max():.1f} q/ha")
            # Split by state and save
            self.split_and_save_maize_data(maize_dataset)

            # Create analysis visualizations
            self.create_maize_analysis_plots(maize_dataset)

            # Create summary report
            self.create_maize_summary_report(maize_dataset)

            print("\n✅ Maize data extraction completed successfully!")
            return maize_dataset

        return None

    def extract_maize_data_from_apy(self):
        """Extract maize data from APY dataset"""
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

            # Filter for maize (check various spellings)
            maize_filters = df['Crop'].str.upper().str.contains('MAIZE|CORN', na=False, regex=True)
            maize_df = df[maize_filters].copy()
            print(f"Maize records found: {len(maize_df)}")

            if len(maize_df) == 0:
                print("⚠️  No maize records found in dataset")
                return None

            # Clean and standardize data
            maize_cleaned = self._preprocess_maize_data(maize_df)

            # Filter for major maize states
            major_maize_df = maize_cleaned[maize_cleaned['State_clean'].isin(self.maize_states)].copy()
            print(f"Filtered to major maize states: {len(major_maize_df)} records")

            if len(major_maize_df) == 0:
                print("⚠️  No records found for major maize states")
                return None

            return major_maize_df

        except Exception as e:
            print(f"❌ Error extracting maize data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _preprocess_maize_data(self, maize_df):
        """Preprocess maize data for analysis"""

        print("🔧 Preprocessing maize data...")

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
        existing_cols = {old: new for old, new in column_mapping.items() if old in maize_df.columns}
        maize_df = maize_df.rename(columns=existing_cols)

        # Standardize state names
        state_mapping = {
            'Karnataka': 'KARNATAKA',
            'Maharashtra': 'MAHARASHTRA',
            'Andhra Pradesh': 'ANDHRA_PRADESH',
            'Bihar': 'BIHAR',
            'Uttar Pradesh': 'UTTAR_PRADESH',
            'Madhya Pradesh': 'MADHYA_PRADESH',
            'Tamil Nadu': 'TAMIL_NADU',
            'Rajasthan': 'RAJASTHAN',
            'Gujarat': 'GUJARAT'
        }

        maize_df['State_clean'] = maize_df['State_clean'].replace(state_mapping)

        # Clean season data
        season_mapping = {
            'Kharif': 'KHARIF',
            'Rabi': 'RABI',
            'Autumn': 'KHARIF',
            'Winter': 'RABI',
            'Summer': 'SUMMER'
        }
        if 'Season_clean' in maize_df.columns:
            maize_df['Season_clean'] = maize_df['Season_clean'].replace(season_mapping)

        # Clean district names
        if 'District_clean' in maize_df.columns:
            maize_df['District_clean'] = maize_df['District_clean'].str.strip()

        # Handle missing values
        maize_df['Area'] = pd.to_numeric(maize_df['Area'], errors='coerce')
        maize_df['Production'] = pd.to_numeric(maize_df['Production'], errors='coerce')
        maize_df['yield_quintal_ha'] = pd.to_numeric(maize_df['yield_quintal_ha'], errors='coerce')

        # Calculate yield where missing
        mask = maize_df['yield_quintal_ha'].isna() & maize_df['Area'].notna() & maize_df['Production'].notna() & (maize_df['Area'] > 0)
        maize_df.loc[mask, 'yield_quintal_ha'] = maize_df.loc[mask, 'Production'] / maize_df.loc[mask, 'Area']

        # Drop rows with no yield data
        maize_df = maize_df.dropna(subset=['yield_quintal_ha', 'Area'])

        # Remove unrealistic yields (maize typically 10-80 q/ha)
        maize_df = maize_df[(maize_df['yield_quintal_ha'] >= 5) & (maize_df['yield_quintal_ha'] <= 120)]

        print(f"Input columns: {list(maize_df.columns)}")
        print(f"Preprocessed maize data: {len(maize_df)} records")

        return maize_df

    def split_and_save_maize_data(self, maize_df):
        """Split maize data by state and save"""

        print("💾 Saving maize datasets...")

        # Save complete dataset
        complete_file = self.main_data_dir / "maize_dataset_complete.csv"
        maize_df.to_csv(complete_file, index=False)
        print(f"Saved complete maize dataset: {len(maize_df)} records")

        # Split by state
        for state in self.maize_states:
            state_data = maize_df[maize_df['State_clean'] == state].copy()
            if len(state_data) > 0:
                state_file = self.main_data_dir / f"maize_{state.lower()}.csv"
                state_data.to_csv(state_file, index=False)
                print(f"Saved {state.lower()} maize data: {len(state_data)} records")

        # Create training and validation splits (temporal)
        if 'Crop_Year' in maize_df.columns:
            training_data = maize_df[maize_df['Crop_Year'] <= 2018].copy()
            validation_data = maize_df[maize_df['Crop_Year'] >= 2019].copy()

            training_file = self.main_data_dir / "maize_training_data.csv"
            validation_file = self.main_data_dir / "maize_validation_data.csv"

            training_data.to_csv(training_file, index=False)
            validation_data.to_csv(validation_file, index=False)

            print(f"Saved training data: {len(training_data)} records (≤2018)")
            print(f"Saved validation data: {len(validation_data)} records (≥2019)")

    def create_maize_analysis_plots(self, maize_df):
        """Create analysis plots for maize data"""

        print("📈 Creating maize data visualizations...")

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Yield distribution by state
            state_yields = maize_df.groupby('State_clean')['yield_quintal_ha'].describe()
            state_yields['mean'].plot(kind='bar', ax=axes[0,0], color='orange')
            axes[0,0].set_title('Average Maize Yield by State (q/ha)')
            axes[0,0].set_ylabel('Yield (q/ha)')
            axes[0,0].tick_params(axis='x', rotation=45)

            # Yield trend over years
            if 'Crop_Year' in maize_df.columns:
                yearly_trend = maize_df.groupby('Crop_Year')['yield_quintal_ha'].mean()
                yearly_trend.plot(ax=axes[0,1], marker='o', color='green')
                axes[0,1].set_title('Maize Yield Trend Over Years')
                axes[0,1].set_ylabel('Average Yield (q/ha)')
                axes[0,1].grid(True, alpha=0.3)

            # Yield distribution histogram
            axes[1,0].hist(maize_df['yield_quintal_ha'], bins=30, alpha=0.7, color='gold', edgecolor='black')
            axes[1,0].set_title('Maize Yield Distribution')
            axes[1,0].set_xlabel('Yield (q/ha)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].axvline(maize_df['yield_quintal_ha'].mean(), color='red', linestyle='--', label='.1f')
            axes[1,0].legend()

            # Area vs Yield scatter
            axes[1,1].scatter(maize_df['Area'], maize_df['yield_quintal_ha'], alpha=0.6, s=30, color='purple')
            axes[1,1].set_title('Maize Area vs Yield Relationship')
            axes[1,1].set_xlabel('Area (hectares)')
            axes[1,1].set_ylabel('Yield (q/ha)')
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plots
            plots_file = self.main_data_dir / "maize_data_analysis.png"
            plt.savefig(plots_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Visualizations saved to {plots_file}")

        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")

    def create_maize_summary_report(self, maize_df):
        """Create comprehensive maize data summary"""

        summary = {
            "extraction_metadata": {
                "extraction_date": datetime.now().isoformat(),
                "source_dataset": "APY.csv",
                "extraction_method": "maize_corn_filtering"
            },
            "dataset_summary": {
                "total_records": len(maize_df),
                "unique_states": len(maize_df['State_clean'].unique()) if 'State_clean' in maize_df.columns else 0,
                "date_range": f"{maize_df['Crop_Year'].min()}-{maize_df['Crop_Year'].max()}" if 'Crop_Year' in maize_df.columns else "unknown",
                "avg_yield": round(maize_df['yield_quintal_ha'].mean(), 2),
                "median_yield": round(maize_df['yield_quintal_ha'].median(), 2),
                "yield_std": round(maize_df['yield_quintal_ha'].std(), 2)
            }
        }

        # State-wise summary
        if 'State_clean' in maize_df.columns:
            state_summary = {}
            for state in maize_df['State_clean'].unique():
                state_data = maize_df[maize_df['State_clean'] == state]
                state_summary[state] = {
                    "record_count": len(state_data),
                    "avg_yield": round(state_data['yield_quintal_ha'].mean(), 2),
                    "max_yield": round(state_data['yield_quintal_ha'].max(), 2),
                    "min_yield": round(state_data['yield_quintal_ha'].min(), 2)
                }

            top_states = sorted(state_summary.items(), key=lambda x: x[1]['record_count'], reverse=True)[:5]
            summary["top_states"] = {state: count for state, stats in top_states for count in [stats['record_count']]}

        # Save summary
        summary_file = self.main_data_dir / "maize_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📋 Maize summary saved: {summary_file}")

        return summary

def main():
    """Main execution"""
    extractor = MaizeDataExtractor()
    return extractor.main()

if __name__ == "__main__":
    main()
