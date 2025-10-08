"""
Rice Data Extraction and Analysis for India Agricultural Intelligence Platform
Extracts rice yield data from APY dataset for multi-state model training
"""

import pandas as pd
import zipfile
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class RiceDataExtractor:
    """Extract and process rice yield data from Indian agricultural datasets"""

    def __init__(self, output_dir="rice_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Major rice-growing states in India
        self.rice_states = [
            'PUNJAB', 'HARYANA', 'UTTAR PRADESH', 'BIHAR', 'MADHYA PRADESH',
            'JHARKHAND', 'RAJASTHAN', 'WEST BENGAL', 'ODISHA', 'ANDHRA PRADESH',
            'TAMIL NADU', 'KARNATAKA', 'ASSAM', 'CHHATTISGARH'
        ]

    def extract_rice_data_from_apy(self):
        """Extract rice data from APY.csv.zip"""

        print("🍚 RICE DATA EXTRACTION - PHASE 1")
        print("=" * 50)

        try:
            # Extract and load APY data
            with zipfile.ZipFile('APY.csv.zip', 'r') as z:
                with z.open('APY.csv') as f:
                    print("Loading APY.csv...")
                    df = pd.read_csv(f)

            print(f"APY Dataset loaded: {len(df)} records")
            print(f"Columns: {list(df.columns)}")

            # Filter for rice data
            rice_df = df[df['Crop'].str.upper() == 'RICE'].copy()
            print(f"\nRice records found: {len(rice_df)}")

            # Analyze rice distribution by state
            state_distribution = rice_df['State'].value_counts()
            print(f"\nRice data distribution by state:")
            for state, count in state_distribution.items():
                print(">12")

            # Filter for major rice states with sufficient data (>100 records)
            major_rice_states = state_distribution[state_distribution > 100].index.tolist()
            major_rice_df = rice_df[rice_df['State'].isin(major_rice_states)].copy()

            print(f"\nFiltered to major rice states: {len(major_rice_df)} records")
            print(f"States included: {major_rice_states}")

            # Clean and preprocess rice data
            rice_cleaned = self._preprocess_rice_data(major_rice_df)

            # Analyze rice yields by state and year
            self._analyze_rice_yields(rice_cleaned)

            # Save processed data
            self._save_rice_datasets(rice_cleaned)

            return rice_cleaned

        except Exception as e:
            print(f"❌ Error extracting rice data: {e}")
            raise

    def _preprocess_rice_data(self, rice_df):
        """Preprocess rice data for ML modeling"""

        print("\n🔧 Preprocessing rice data...")
        print(f"Input columns: {list(rice_df.columns)}")

        # Fix column names (handle trailing spaces)
        column_mapping = {
            'Area ': 'Area',
            'District ': 'District',
            'Crop_Year': 'Crop_Year',
            'Yield': 'Yield',
            'Production': 'Production'
        }

        rice_df = rice_df.rename(columns=column_mapping)

        # Convert yield to quintal/ha (standard unit)
        rice_df['yield_quintal_ha'] = rice_df['Yield'] / 10  # kg/ha to quintal/ha

        # Handle missing values - use actual column names
        rice_df = rice_df.dropna(subset=['Yield', 'Area', 'Production'])

        # Add derived features
        rice_df['yield_efficiency'] = rice_df['Production'] / rice_df['Area']  # tonnes/ha
        rice_df['area_category'] = pd.cut(rice_df['Area'],
                                         bins=[0, 100, 500, 1000, float('inf')],
                                         labels=['Small', 'Medium', 'Large', 'Very Large'])

        # Clean state names
        rice_df['State_clean'] = rice_df['State'].str.strip().str.upper()

        # Convert season (Rabi/Kharif)
        rice_df['Season_clean'] = rice_df['Season'].str.strip().str.upper()

        # Add irrigation proxy (based on historical knowledge)
        rice_states_irrigation = {
            'PUNJAB': 0.95,
            'HARYANA': 0.88,
            'UTTAR PRADESH': 0.72,
            'BIHAR': 0.45,
            'WEST BENGAL': 0.55,
            'ODISHA': 0.35,
            'ANDHRA PRADESH': 0.65,
            'TAMIL NADU': 0.58,
            'KARNATAKA': 0.25,
            'ASSAM': 0.15
        }

        rice_df['irrigation_coverage'] = rice_df['State_clean'].map(rice_states_irrigation).fillna(0.5)

        print(f"Preprocessed rice data: {len(rice_df)} records")
        print(f"Date range: {rice_df['Crop_Year'].min()} - {rice_df['Crop_Year'].max()}")
        print(".2f")

        return rice_df

    def _analyze_rice_yields(self, rice_df):
        """Analyze rice yield patterns across states and time"""

        print("\n📊 Analyzing rice yield patterns...")

        # Yield statistics by state
        yield_by_state = rice_df.groupby('State_clean')['yield_quintal_ha'].agg(['mean', 'std', 'min', 'max', 'count'])
        yield_by_state = yield_by_state.round(2)

        print(f"\nRice yield statistics by state (quintal/ha):")
        print(yield_by_state)

        # Yield trends over time
        yield_trends = rice_df.groupby(['Crop_Year', 'State_clean'])['yield_quintal_ha'].mean().unstack()
        recent_years = yield_trends.tail(5)  # Last 5 years

        print(f"\nRecent rice yield trends (last 5 years):")
        print(recent_years.round(1))

        # Seasonal analysis
        seasonal_yields = rice_df.groupby(['Season_clean', 'State_clean'])['yield_quintal_ha'].agg(['mean', 'count'])
        seasonal_yields = seasonal_yields.round(2)

        print(f"\nRice yields by season and state:")
        print(seasonal_yields.head(20))  # First 20 entries

    def _save_rice_datasets(self, rice_df):
        """Save processed rice datasets for model training"""

        print("\n💾 Saving rice datasets...")

        # Save full rice dataset
        rice_df.to_csv(self.output_dir / "rice_dataset_complete.csv", index=False)

        # Split by state for multi-state modeling
        for state in rice_df['State_clean'].unique():
            state_data = rice_df[rice_df['State_clean'] == state].copy()

            if len(state_data) >= 50:  # Minimum records for modeling
                state_filename = f"rice_{state.lower().replace(' ', '_')}.csv"
                state_data.to_csv(self.output_dir / state_filename, index=False)
                print(f"Saved {state} rice data: {len(state_data)} records")

        # Training/validation splits
        recent_years = rice_df[rice_df['Crop_Year'] >= 2010]  # Last decade training data

        # Train/test split by year (temporal validation)
        train_years = recent_years[recent_years['Crop_Year'] <= 2018]
        test_years = recent_years[recent_years['Crop_Year'] >= 2019]

        train_years.to_csv(self.output_dir / "rice_training_data.csv", index=False)
        test_years.to_csv(self.output_dir / "rice_validation_data.csv", index=False)

        print(f"\nTraining data: {len(train_years)} records (≤2018)")
        print(f"Validation data: {len(test_years)} records (≥2019)")

        # Summary statistics
        summary = {
            'total_records': len(rice_df),
            'unique_states': len(rice_df['State_clean'].unique()),
            'date_range': f"{rice_df['Crop_Year'].min()}-{rice_df['Crop_Year'].max()}",
            'avg_yield': round(rice_df['yield_quintal_ha'].mean(), 2),
            'median_yield': round(rice_df['yield_quintal_ha'].median(), 2),
            'yield_std': round(rice_df['yield_quintal_ha'].std(), 2),
            'top_states': rice_df['State_clean'].value_counts().head(5).to_dict()
        }

        with open(self.output_dir / "rice_data_summary.json", 'w') as f:
            import json
            json.dump(summary, f, indent=2)

        print("\n📋 Rice dataset summary:")
        for key, value in summary.items():
            print(">20")

    def create_rice_visualizations(self, rice_df):
        """Create visualizations for rice data analysis"""

        print("\n📈 Creating rice data visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Rice yield distribution by state
        plt.figure(figsize=(12, 8))
        states_sorted = rice_df.groupby('State_clean')['yield_quintal_ha'].mean().sort_values(ascending=False)
        states_sorted.head(10).plot(kind='bar', color='skyblue')
        plt.title('Average Rice Yields by State (Top 10)', fontsize=14, fontweight='bold')
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Yield (Quintal/ha)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "rice_yields_by_state.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Rice yield trends over time (top states)
        plt.figure(figsize=(14, 8))
        top_states = rice_df.groupby('State_clean')['yield_quintal_ha'].count().nlargest(5).index

        for state in top_states:
            state_data = rice_df[rice_df['State_clean'] == state]
            yearly_yields = state_data.groupby('Crop_Year')['yield_quintal_ha'].mean()
            plt.plot(yearly_yields.index, yearly_yields.values, marker='o', linewidth=2, label=state)

        plt.title('Rice Yield Trends Over Time (Top 5 States)', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Yield (Quintal/ha)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "rice_yield_trends.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Rice area vs yield scatter plot
        plt.figure(figsize=(10, 6))
        # Sample data to avoid overplotting
        sample_data = rice_df.sample(min(5000, len(rice_df)))
        plt.scatter(sample_data['Area'], sample_data['yield_quintal_ha'],
                   alpha=0.6, color='green', s=30)
        plt.title('Rice Area vs Yield Relationship (Sample)', fontsize=14, fontweight='bold')
        plt.xlabel('Area (hectares)', fontsize=12)
        plt.ylabel('Yield (Quintal/ha)', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / "rice_area_yield_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Visualizations saved to rice_data/ directory")

def main():
    """Main execution function"""
    print("🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM")
    print("🍚 Rice Data Extraction for Multi-State Model Training")
    print("=" * 60)

    extractor = RiceDataExtractor()

    try:
        # Extract and process rice data
        rice_data = extractor.extract_rice_data_from_apy()

        # Create visualizations
        extractor.create_rice_visualizations(rice_data)

        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Rice data extraction completed!")
        print("=" * 60)

        # Display key metrics
        summary_file = extractor.output_dir / "rice_data_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            print("📊 KEY METRICS:")
            print(f"• Total Rice Records: {summary['total_records']:,}")
            print(f"• States Covered: {summary['unique_states']}")
            print(".1f")
            print(".1f")
            print(f"• Date Range: {summary['date_range']}")
            print(f"• Top Producing States: {', '.join(list(summary['top_states'].keys())[:3])}")

        print("\n📁 Generated files:")
        for file in extractor.output_dir.glob("*.csv"):
            print(f"• {file.name} ({file.stat().st_size / 1024:.0f} KB)")
        for file in extractor.output_dir.glob("*.png"):
            print(f"• {file.name}")

        return rice_data

    except Exception as e:
        print(f"❌ ERROR: Rice data extraction failed: {e}")
        raise

if __name__ == "__main__":
    rice_dataset = main()
