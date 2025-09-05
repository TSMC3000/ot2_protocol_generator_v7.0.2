import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import from the existing modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath('.'))
from matal.utils import get_sid

# Page config
st.set_page_config(page_title="OT2 Protocol Generator", layout="wide")
st.title("OT2 Protocol Generator")

# Initialize session state
if 'composition_data' not in st.session_state:
    st.session_state.composition_data = None
if 'batch_config' not in st.session_state:
    st.session_state.batch_config = {}

# All tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Composition Input",
    "2. Batch Configuration", 
    "3. Source Configuration",
    "4. Sample Calculation",
    "5. Protocol Configuration",
    "6. Generate & Download"
])

# Tab 1: Composition Input
with tab1:
    st.header("Upload Composition Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the original dataframe
        df_original = pd.read_csv(uploaded_file)
        
        # Check if this is a new file by comparing filename or content
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Reset selections if it's a new file
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_identifier:
            st.session_state.current_file_id = file_identifier
            # Reset all related session states
            if 'selected_columns' in st.session_state:
                del st.session_state.selected_columns
            if 'source_order' in st.session_state:
                del st.session_state.source_order
            if 'selected_source_idx' in st.session_state:
                del st.session_state.selected_source_idx
            # You can add more session state resets here if needed
            st.info("📄 New file detected - column selections have been reset")
        
        # Column Selection Section
        st.subheader("Column Selection")
        
        # Get all columns
        all_columns = df_original.columns.tolist()
        
        # Identify potential ID columns and material columns
        id_columns = [col for col in all_columns if col in ['SampleID', 'Unnamed: 0'] or 'id' in col.lower()]
        material_columns = [col for col in all_columns if col not in id_columns]
        
        # Initialize selected columns in session state if not exists
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = material_columns.copy()
        
        # Ensure selected_columns only contains valid columns from current file
        st.session_state.selected_columns = [col for col in st.session_state.selected_columns if col in material_columns]
        
        # Calculate excluded columns
        excluded_columns = [col for col in material_columns if col not in st.session_state.selected_columns]
        
        # Create two columns for interactive selection
        select_col, exclude_col = st.columns(2)
        
        with select_col:
            st.write("**Select columns to use (ID and unnamed columns have been excluded automatically):**")
            
            # Display selected columns as clickable badges
            if st.session_state.selected_columns:
                # Create a container for the badges
                for i in range(0, len(st.session_state.selected_columns), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        if i + j < len(st.session_state.selected_columns):
                            col_name = st.session_state.selected_columns[i + j]
                            with cols[j]:
                                if st.button(
                                    col_name,
                                    key=f"selected_{col_name}",
                                    help=f"Click to exclude {col_name}",
                                    use_container_width=True
                                ):
                                    # Move to excluded
                                    st.session_state.selected_columns.remove(col_name)
                                    st.rerun()
            else:
                st.info("No columns selected. Click excluded columns to add them.")
        
        with exclude_col:
            st.write("**Excluded column names:**")
            
            # Display excluded columns as clickable badges
            if excluded_columns:
                # Create a container for the badges
                for i in range(0, len(excluded_columns), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        if i + j < len(excluded_columns):
                            col_name = excluded_columns[i + j]
                            with cols[j]:
                                if st.button(
                                    col_name,
                                    key=f"excluded_{col_name}",
                                    help=f"Click to include {col_name}",
                                    use_container_width=True
                                ):
                                    # Move to selected
                                    st.session_state.selected_columns.append(col_name)
                                    st.rerun()
            else:
                st.success("No columns excluded. All materials will be used.")
        
        # Apply column selection
        selected_cols = st.session_state.selected_columns
        df = df_original[selected_cols].copy() if selected_cols else df_original.copy()
        st.session_state.composition_data = df
        
        st.divider()
        
        # Data Preview
        st.subheader("Data Preview (After Column Selection)")
        st.dataframe(df)
        
        # Validation
        st.subheader("Validation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Materials Selected:**")
            materials = selected_cols
            if materials:
                st.metric("Total Selected", len(materials))
                for mat in materials[:5]:  # Show first 5 for space
                    st.write(f"✓ {mat}")
                if len(materials) > 5:
                    st.write(f"... and {len(materials)-5} more")
            else:
                st.error("No materials selected!")
        
        with col2:
            st.write("**Composition Sum Check:**")
            if selected_cols:
                # Only check numeric columns from selected columns
                numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns]
                
                if numeric_cols:
                    sums = df[numeric_cols].sum(axis=1)
                    
                    issues = []
                    for idx, sum_val in enumerate(sums):
                        if abs(sum_val - 1.0) > 0.01:
                            issues.append(f"Row {idx}: Sum = {sum_val:.4f}")
                    
                    if issues:
                        st.warning("⚠️ Some rows don't sum to 1.0:")
                        for issue in issues[:5]:  # Show first 5 issues
                            st.write(issue)
                        if len(issues) > 5:
                            st.write(f"... and {len(issues)-5} more rows")
                        st.info("Consider adjusting your column selection or data values")
                    else:
                        st.success("✓ All compositions sum to 1.0")
                else:
                    st.error("No numeric columns found in selection")
            else:
                st.error("Please select at least one column")

# Tab 2: Batch Configuration  
with tab2:
    st.header("Batch Configuration")
    
    if st.session_state.composition_data is None:
        st.warning("Please upload composition data in Tab 1 first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration Input")
            
            proj_code = st.text_input("Project Code (2 chars: lowercase + uppercase)", 
                                      value="aT", 
                                      max_chars=2,
                                      help="First character lowercase, second uppercase")
            
            if len(proj_code) == 2:
                if not (proj_code[0].islower() and proj_code[1].isupper()):
                    st.error("Format must be: lowercase + uppercase (e.g., aT)")
            
            # Item type selection
            item_types = ['Batch', 'Sample', 'OT2Protocol']
            item_type = st.selectbox("Item Type", options=item_types, index=0)
            
            # Usage type for get_sid function
            usage_options = [
                'test_data',
                'optimization',
                'active_learning',
                'active_learning_trial',
                'miscellaneous',
                'design_boundary',
                'design_boundary_opt',
                'design_boundary_variance',
                'revision_exp'
            ]
            usage = st.selectbox("Usage Type", options=usage_options, index=0)
            
            batch_num = st.number_input("Batch Number/Index", min_value=0, value=0)
        
        with col2:
            st.subheader("Generated SID")
            
            if len(proj_code) == 2 and proj_code[0].islower() and proj_code[1].isupper():
                try:
                    if item_type == 'Batch':
                        generated_sid = get_sid('Batch', i=batch_num, usage=usage, proj_code=proj_code)
                    elif item_type == 'Sample':
                        # Need batch_sid for Sample generation
                        batch_sid = get_sid('Batch', i=batch_num, usage=usage, proj_code=proj_code)
                        generated_sid = get_sid('Sample', batch_sid=batch_sid, i=1)
                    else:  # OT2Protocol
                        batch_sid = get_sid('Batch', i=batch_num, usage=usage, proj_code=proj_code)
                        generated_sid = get_sid('OT2Protocol', batch_sid=batch_sid, i=1)
                    
                    st.session_state.batch_config = {
                        'batch_sid': generated_sid if item_type == 'Batch' else batch_sid,
                        'generated_sid': generated_sid,
                        'batch_num': batch_num,
                        'proj_code': proj_code,
                        'item_type': item_type,
                        'usage': usage
                    }
                    
                    st.success(f"**Generated {item_type} SID: {generated_sid}**")
                    
                    # Show editable parts based on item type
                    st.info(f"""
                    **Editable Parts for {item_type}:**
                    - Project Code: {proj_code}
                    - Item Type: {item_type}
                    - Usage: {usage}
                    - Index: {batch_num}
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating SID: {str(e)}")
            else:
                st.error("Please enter valid project code to generate SID")
        
        if 'batch_sid' in st.session_state.batch_config:
            st.divider()
            st.write("**Current Configuration:**")
            st.json(st.session_state.batch_config)

# Tab 3: Source Configuration
with tab3:
    st.header("Source Configuration")
    
    if st.session_state.composition_data is None:
        st.warning("Please upload composition data in Tab 1 first")
    elif 'batch_sid' not in st.session_state.batch_config:
        st.warning("Please configure batch in Tab 2 first")
    else:
        # Extract unique materials from composition data
        materials = [col for col in st.session_state.composition_data.columns if col != 'SampleID']
        
        st.subheader("Material Concentration Settings")
        
        # Initialize source config in session state
        if 'source_config' not in st.session_state:
            st.session_state.source_config = {}
        
        # Add any new materials that aren't in source_config yet
        for mat in materials:
            if mat not in st.session_state.source_config:
                st.session_state.source_config[mat] = {
                    'material_id': f"{mat}-0000",
                    'concentration': 10.0
                }
        
        # Remove any materials from source_config that aren't in the current CSV
        materials_to_remove = [mat for mat in st.session_state.source_config.keys() if mat not in materials]
        for mat in materials_to_remove:
            del st.session_state.source_config[mat]
        
        # Bulk concentration setter
        col1, col2 = st.columns([1, 3])
        with col1:
            bulk_conc = st.number_input("Set All Concentrations (mg/mL)", 
                                       min_value=0.1, 
                                       value=10.0, 
                                       step=0.1)
            if st.button("Apply to All"):
                for mat in materials:
                    st.session_state.source_config[mat]['concentration'] = bulk_conc
                st.rerun()
        
        # Create editable dataframe
        source_data = []
        for mat in materials:
            source_data.append({
                'Material Code': mat,
                'Material ID': st.session_state.source_config[mat]['material_id'],
                'Concentration (mg/mL)': st.session_state.source_config[mat]['concentration']
            })
        
        df_sources = pd.DataFrame(source_data)
        
        # Display editable table
        edited_df = st.data_editor(
            df_sources,
            column_config={
                "Material Code": st.column_config.TextColumn(disabled=True),
                "Material ID": st.column_config.TextColumn(disabled=True),
                "Concentration (mg/mL)": st.column_config.NumberColumn(
                    min_value=0.1,
                    step=0.1,
                    format="%.1f"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Update session state with edited values
        for idx, row in edited_df.iterrows():
            mat = row['Material Code']
            st.session_state.source_config[mat]['concentration'] = row['Concentration (mg/mL)']
        
        # Display summary
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Materials", len(materials))
        with col2:
            avg_conc = sum(s['concentration'] for s in st.session_state.source_config.values()) / len(materials)
            st.metric("Average Concentration", f"{avg_conc:.1f} mg/mL")
        
        # Show current configuration
        if st.checkbox("Show Source Configuration Details"):
            st.json(st.session_state.source_config)

# Tab 4: Sample Calculation
with tab4:
    st.header("Sample Calculation")
    
    if st.session_state.composition_data is None:
        st.warning("Please upload composition data in Tab 1 first")
    elif 'batch_sid' not in st.session_state.batch_config:
        st.warning("Please configure batch in Tab 2 first")
    elif 'source_config' not in st.session_state:
        st.warning("Please configure sources in Tab 3 first")
    else:
        st.subheader("Calculation Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            target_mass = st.number_input("Target Mass (mg)", min_value=1, value=120)
            max_sources = st.number_input("Max Sources per Sample", min_value=1, value=5)
        with col2:
            min_volume = st.number_input("Min Volume Threshold (µL)", min_value=0, value=100)
            st.info("Volumes < threshold: set to threshold if ≥50µL, else 0")
        
        if st.button("Calculate Samples"):
            # Prepare data structures
            batch_sid = st.session_state.batch_config['batch_sid']
            ONE = 10000
            sample_list = []
            
            # Create code lookup table
            code_lut = {mat: config['material_id'] for mat, config in st.session_state.source_config.items()}
            cons_lut = {config['material_id']: config['concentration'] for config in st.session_state.source_config.values()}
            
            # Process each sample
            # In the "Calculate Samples" button section, update the loop:
            for idx, row in st.session_state.composition_data.iterrows():
                line = {'sample_sid': get_sid('Sample', batch_sid, idx + 1)}
                
                n_srcs = 0
                for m_type, m_frac in row.items():
                    # Skip any ID columns
                    if m_type in ['SampleID', 'Unnamed: 0']:
                        continue
                    
                    m_code = code_lut[m_type]
                    if m_frac > 0:
                        vol = round(target_mass * m_frac / cons_lut[m_code] * 1000)
                        
                        # Apply minimum volume threshold
                        if vol < min_volume:
                            vol = min_volume if vol >= 50 else 0
                        
                        if vol > 0:
                            n_srcs += 1
                            line[f'src{n_srcs}_sid'] = m_code
                            line[f'src{n_srcs}_frc'] = round(vol / 1000 * cons_lut[m_code] / target_mass * ONE)
                            line[f'src{n_srcs}_mas'] = vol / 1000 * cons_lut[m_code]
                            line[f'src{n_srcs}_vol'] = vol
                            
                            # Fill remaining source slots with NULL
                            for i in range(n_srcs + 1, max_sources + 1):
                                line[f'src{i}_sid'] = 'NUL-0000'
                                line[f'src{i}_frc'] = 0
                                line[f'src{i}_mas'] = 0
                                line[f'src{i}_vol'] = 0
                
                # Calculate actual total mass and scale
                total_mass_actual = sum(line[f'src{i}_mas'] for i in range(1, max_sources + 1))
                
                # Recalculate fractions based on actual mass
                for i in range(1, max_sources + 1):
                    if total_mass_actual > 0:
                        line[f'src{i}_frc'] = round(line[f'src{i}_mas'] / total_mass_actual * ONE)
                
                line['mass'] = round(total_mass_actual, 4)
                line['additional_flags'] = ''
                sample_list.append(line)
            
            # Store calculated samples
            st.session_state.calculated_samples = pd.DataFrame(sample_list)
            st.success(f"Calculated {len(sample_list)} samples")
        
        # Display results if available
        if 'calculated_samples' in st.session_state:
            st.subheader("Calculated Samples Preview")
            
            # Summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", len(st.session_state.calculated_samples))
            with col2:
                avg_mass = st.session_state.calculated_samples['mass'].mean()
                st.metric("Average Mass", f"{avg_mass:.2f} mg")
            
            # Show sample details with all source info
            display_cols = ['sample_sid', 'mass']
            for i in range(1, max_sources + 1):
                display_cols.extend([
                    f'src{i}_sid',
                    f'src{i}_frc', 
                    f'src{i}_mas',
                    f'src{i}_vol'
                ])
            
            # Format the dataframe for better display
            df_display = st.session_state.calculated_samples[display_cols].copy()
            
            # Format fraction columns as percentages
            for i in range(1, max_sources + 1):
                df_display[f'src{i}_frc'] = df_display[f'src{i}_frc'] / 100  # Convert from ONE scale to percentage
            
            st.dataframe(
                df_display,
                column_config={
                    'sample_sid': 'Sample ID',
                    'mass': st.column_config.NumberColumn('Total Mass (mg)', format="%.4f"),
                    **{f'src{i}_sid': f'Source {i} ID' for i in range(1, max_sources + 1)},
                    **{f'src{i}_frc': st.column_config.NumberColumn(f'Src {i} Frac (%)', format="%.1f") 
                       for i in range(1, max_sources + 1)},
                    **{f'src{i}_mas': st.column_config.NumberColumn(f'Src {i} Mass (mg)', format="%.4f") 
                       for i in range(1, max_sources + 1)},
                    **{f'src{i}_vol': st.column_config.NumberColumn(f'Src {i} Vol (µL)', format="%.0f") 
                       for i in range(1, max_sources + 1)},
                },
                use_container_width=True,
                height=400
            )
            
            # Verification warnings
            issues = st.session_state.calculated_samples[
                abs(st.session_state.calculated_samples['mass'] - target_mass) > target_mass * 0.1
            ]
            if len(issues) > 0:
                st.warning(f"⚠️ {len(issues)} samples deviate >10% from target mass")
    
# Tab 5: Protocol Configuration
with tab5:
    st.header("Protocol Configuration")
    
    if 'calculated_samples' not in st.session_state:
        st.warning("Please calculate samples in Tab 4 first")
    else:
        # Initialize protocol config in session state
        if 'protocol_config' not in st.session_state:
            st.session_state.protocol_config = {}
        
        # Basic Settings
        st.subheader("Basic Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            proj_owner = st.text_input("Project Owner (2 uppercase chars)", value="YL", max_chars=2)
            if len(proj_owner) == 2 and not proj_owner.isupper():
                st.error("Must be 2 uppercase characters")
        with col2:
            batch_desc = st.text_input("Batch Description", value="LARGE membrane")
        with col3:
            samples_per_protocol = st.number_input("Samples per Protocol", min_value=1, value=30)
        
        # Deck Configuration
        st.subheader("Deck Configuration")
        col1, col2 = st.columns(2)
        with col1:
            src_slots = st.multiselect("Source Slots", 
                                      options=list(range(2, 12)), 
                                      default=[2, 3, 4, 5, 6])
            dst_slots = st.multiselect("Destination Slots", 
                                      options=list(range(2, 12)), 
                                      default=[7, 8, 9, 10, 11])
            tip_slot = st.selectbox("Main Tip Slot", options=[1, 2], index=0)
        
        with col2:
            # Labware types
            labware_options = ['OT_TUBE_50ML', 'OT_TUBE_15ML', 'OT_TUBE_1.5ML', 'OT_TUBE_2ML']
            src_labware = st.selectbox("Source Labware Type", options=labware_options, index=0)
            dst_labware = st.selectbox("Destination Labware Type", options=labware_options, index=0)
            
            # Tip labware types
            tip_options = ['OT_TIP_1000uL', 'OT_TIP_300uL', 'OT_TIP_20uL']
            tip_labware = st.selectbox("Main Tip Type (p1000)", options=tip_options, index=0)
        
        # Validate slot conflicts
        if set(src_slots) & set(dst_slots):
            st.error("Source and destination slots cannot overlap!")
        if tip_slot in src_slots or tip_slot in dst_slots:
            st.error("Tip slot conflicts with source or destination slots!")
        
        # Liquid Handling
        st.subheader("Liquid Handling")
        col1, col2 = st.columns(2)
        
        with col1:
            water_addition = st.checkbox("Add Water")
            if water_addition:
                water_target_vol = st.number_input("Water Target Volume (µL)", min_value=0, value=0)
            else:
                water_target_vol = 0
            
            pipetting_delays = st.checkbox("Enable Pipetting Delays")
            if pipetting_delays:
                aspirate_delay = st.number_input("Aspirate Delay (seconds)", min_value=0, value=10)
                airgap_delay = st.number_input("Airgap Delay (seconds)", min_value=0, value=5)
                dispense_delay = st.number_input("Dispense Delay (seconds)", min_value=0, value=5)
                blowout_delay = st.number_input("Blowout Delay (seconds)", min_value=0, value=10)
                
                # Materials that skip delays
                if 'source_config' in st.session_state:
                    material_ids = [config['material_id'] for config in st.session_state.source_config.values()]
                    skip_delay_materials = st.multiselect(
                        "Materials that skip delays:",
                        options=material_ids,
                        default=[],
                        help="Select materials that should skip pipetting delays"
                    )
                else:
                    skip_delay_materials = []
            else:
                aspirate_delay = airgap_delay = dispense_delay = blowout_delay = 0
                skip_delay_materials = []
        
        with col2:
            alt_pipette = st.checkbox("Use Alternative Pipette")
            if alt_pipette:
                alt_tip_slot = st.selectbox("Alternative Tip Slot", options=[1, 2], index=1)
                alt_tip_labware = st.selectbox("Alt Tip Type (p300)", options=tip_options, index=1)
                if alt_tip_slot in src_slots or alt_tip_slot in dst_slots or alt_tip_slot == tip_slot:
                    st.error("Alternative pipette slot conflicts with other slots!")
            else:
                alt_tip_slot = 2
                alt_tip_labware = 'OT_TIP_300uL'
        
        # Processing Control
        st.subheader("Processing Control")
        
        col1, col2 = st.columns(2)
        with col1:
            pause_before = st.checkbox("Pause Before Source")
            if pause_before and 'source_config' in st.session_state:
                material_ids = [config['material_id'] for config in st.session_state.source_config.values()]
                pause_before_src = st.selectbox("Select Source to Pause Before", 
                                               options=['None'] + material_ids,
                                               key="pause_before_select")
                if pause_before_src == 'None':
                    pause_before_src = []
            else:
                pause_before_src = []
            
            starting_index = st.number_input("Starting Protocol Index", min_value=0, value=0)
        
        with col2:
            pause_after = st.checkbox("Pause After Source")
            if pause_after and 'source_config' in st.session_state:
                material_ids = [config['material_id'] for config in st.session_state.source_config.values()]
                pause_after_src = st.selectbox("Select Source to Pause After",
                                              options=['None'] + material_ids,
                                              key="pause_after_select")
                if pause_after_src == 'None':
                    pause_after_src = []
            else:
                pause_after_src = []
        
        # Source Order Management with Improved UI
        if 'source_config' in st.session_state:
            st.subheader("Source Order Management")
            
            material_ids = [config['material_id'] for config in st.session_state.source_config.values()]
            
            # Initialize source order and selected index in session state
            if 'source_order' not in st.session_state:
                st.session_state.source_order = material_ids.copy()
            
            if 'selected_source_idx' not in st.session_state:
                st.session_state.selected_source_idx = 0
            
            # Ensure selected index is within bounds
            if st.session_state.selected_source_idx >= len(st.session_state.source_order):
                st.session_state.selected_source_idx = len(st.session_state.source_order) - 1
            
            # Create two columns for source order display and controls
            order_col, control_col = st.columns([2, 1])
            
            with order_col:
                st.write("**Current Source Order:**")
                
                # Display as a grid with visual selection indicator
                num_cols = 3  # Number of columns in the grid
                rows = len(st.session_state.source_order) // num_cols + (1 if len(st.session_state.source_order) % num_cols else 0)
                
                for row_idx in range(rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        idx = row_idx * num_cols + col_idx
                        if idx < len(st.session_state.source_order):
                            with cols[col_idx]:
                                # Create button for each source with visual selection
                                is_selected = idx == st.session_state.selected_source_idx
                                # Always show order number for ALL items, add arrow only for selected
                                if is_selected:
                                    button_label = f"▶ **{idx+1}. {st.session_state.source_order[idx]}**"
                                else:
                                    button_label = f"**{idx+1}.** {st.session_state.source_order[idx]}"
                                
                                if st.button(
                                    button_label,
                                    key=f"src_btn_{idx}",
                                    type="primary" if is_selected else "secondary",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_source_idx = idx
                                    st.rerun()
            
            with control_col:
                st.write("**Move Selected Item:**")
                
                selected_idx = st.session_state.selected_source_idx
                selected_item = st.session_state.source_order[selected_idx] if st.session_state.source_order else None
                
                if selected_item:
                    st.info(f"Selected: **{selected_item}**")
                    st.write(f"Position: {selected_idx + 1} of {len(st.session_state.source_order)}")
                    
                    # Movement buttons in a 2x2 grid
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        # Move to top
                        if st.button("⬆⬆ Top", use_container_width=True, disabled=(selected_idx == 0)):
                            item = st.session_state.source_order.pop(selected_idx)
                            st.session_state.source_order.insert(0, item)
                            st.session_state.selected_source_idx = 0
                            st.rerun()
                        
                        # Move up one position
                        if st.button("⬆ Up", use_container_width=True, disabled=(selected_idx == 0)):
                            st.session_state.source_order[selected_idx], st.session_state.source_order[selected_idx-1] = \
                                st.session_state.source_order[selected_idx-1], st.session_state.source_order[selected_idx]
                            st.session_state.selected_source_idx = selected_idx - 1
                            st.rerun()
                    
                    with btn_col2:
                        # Move to bottom
                        if st.button("⬇⬇ Bottom", use_container_width=True, disabled=(selected_idx == len(st.session_state.source_order) - 1)):
                            item = st.session_state.source_order.pop(selected_idx)
                            st.session_state.source_order.append(item)
                            st.session_state.selected_source_idx = len(st.session_state.source_order) - 1
                            st.rerun()
                        
                        # Move down one position
                        if st.button("⬇ Down", use_container_width=True, disabled=(selected_idx == len(st.session_state.source_order) - 1)):
                            st.session_state.source_order[selected_idx], st.session_state.source_order[selected_idx+1] = \
                                st.session_state.source_order[selected_idx+1], st.session_state.source_order[selected_idx]
                            st.session_state.selected_source_idx = selected_idx + 1
                            st.rerun()
                    
                    # Quick jump controls
                    st.divider()
                    st.write("**Quick Jump:**")
                    target_pos = st.number_input(
                        "Move to position:",
                        min_value=1,
                        max_value=len(st.session_state.source_order),
                        value=selected_idx + 1,
                        key="jump_position"
                    )
                    if st.button("Jump", use_container_width=True):
                        if target_pos - 1 != selected_idx:
                            item = st.session_state.source_order.pop(selected_idx)
                            st.session_state.source_order.insert(target_pos - 1, item)
                            st.session_state.selected_source_idx = target_pos - 1
                            st.rerun()
                    
                    # Reset button
                    st.divider()
                    if st.button("🔄 Reset Order", use_container_width=True):
                        st.session_state.source_order = material_ids.copy()
                        st.session_state.selected_source_idx = 0
                        st.rerun()
            
            # Display final order as a list
            with st.expander("View Final Source Order"):
                for i, mat in enumerate(st.session_state.source_order):
                    st.write(f"{i+1}. {mat}")
        
        # Save configuration
        st.divider()
        if st.button("Save Protocol Configuration", type="primary", use_container_width=True):
            st.session_state.protocol_config = {
                'proj_code': st.session_state.batch_config.get('proj_code', 'aT'),
                'proj_owner': proj_owner,
                'batch_sid': st.session_state.batch_config.get('batch_sid'),
                'batch_desc': batch_desc,
                'add_water': water_addition,
                'water_target_vol': water_target_vol,
                'src_slots': ','.join(map(str, src_slots)),
                'dst_slots': ','.join(map(str, dst_slots)),
                'tip_slots': str(tip_slot),
                'tip_labware_type': tip_labware,
                'alt_tip_labware_type': alt_tip_labware,
                'src_labware_type': src_labware,
                'dst_labware_type': dst_labware,
                'n_samples_per_proc': samples_per_protocol,
                'scaling_factor': 1.0,
                'aspirate_delay': aspirate_delay,
                'airgap_delay': airgap_delay,
                'dispense_delay': dispense_delay,
                'blowout_delay': blowout_delay,
                'skip_delay_materials': skip_delay_materials if pipetting_delays else [],
                'use_alt_pipette': alt_pipette,
                'alt_tip_slots': str(alt_tip_slot),
                'src_order': st.session_state.source_order if 'source_order' in st.session_state else [],
                'pause_before_src': pause_before_src if pause_before else [],
                'pause_after_src': pause_after_src if pause_after else [],
                'proc_sid_start': starting_index,
                'water_first': False,
                'label_template': 'avery_8167',
                'label_types': 'ot2_src,ot2_dst,oven_boat,dish_comp_barcode'
            }
            
            # Add sources configuration
            if 'source_config' in st.session_state:
                st.session_state.protocol_config['sources'] = {
                    config['material_id']: {'concentration': f"{config['concentration']:.1f} mg/mL"}
                    for config in st.session_state.source_config.values()
                }
            
            st.success("✅ Protocol configuration saved!")
        
        # Display current configuration
        if st.session_state.protocol_config:
            with st.expander("View Current Configuration"):
                st.json(st.session_state.protocol_config)
       
# Tab 6: Generate & Download
with tab6:
    st.header("Generate & Download")
    
    if 'protocol_config' not in st.session_state:
        st.warning("Please complete all previous steps first")
    else:
        st.subheader("Generate OT2 Protocol")
        
        if st.button("Generate Protocol", type="primary"):
            try:
                with st.spinner("Generating protocol files..."):
                    # Import OT2ProcGeneratorV7
                    from matal.robot.ot2 import OT2ProcGeneratorV7
                    
                    # Create generator instance
                    generator = OT2ProcGeneratorV7(
                        st.session_state.calculated_samples,
                        configs=st.session_state.protocol_config
                    )
                    
                    # Generate protocols
                    proc_sids = generator.generate()
                    
                    # Store generator in session state for later use
                    st.session_state.generator = generator
                    st.session_state.generated_protocols = proc_sids
                    st.success(f"Successfully generated {len(proc_sids)} protocol(s)!")
                    
            except Exception as e:
                st.error(f"Error generating protocol: {str(e)}")
        
        # Display generated protocols and download links
        if 'generated_protocols' in st.session_state and 'generator' in st.session_state:
            st.subheader("Generated Files")
            
            generator = st.session_state.generator  # Retrieve generator from session state
            
            for proc_sid in st.session_state.generated_protocols:
                st.write(f"**Protocol: {proc_sid}**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Main protocol file
                    json_path = generator.get_asset_path(proc_sid, 'ot2_proc')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            json_content = f.read()
                        st.download_button(
                            label="📄 OT2 Protocol JSON",
                            data=json_content,
                            file_name=f"{proc_sid}.ot2_proc.json",
                            mime="application/json"
                        )
                
                with col2:
                    # Labels PDF
                    pdf_path = generator.get_asset_path(proc_sid, 'labels')
                    if os.path.exists(pdf_path):
                        with open(pdf_path, 'rb') as f:
                            pdf_content = f.read()
                        st.download_button(
                            label="🏷️ Labels PDF",
                            data=pdf_content,
                            file_name=f"{proc_sid}.labels.pdf",
                            mime="application/pdf"
                        )
                
                with col3:
                    # Debug files as ZIP
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Add all debug files
                        debug_files = [
                            'debug_ot2_steps',
                            'debug_ot2_wells', 
                            'debug_sources',
                            'debug_samples',
                            'debug_configs'
                        ]
                        
                        for file_type in debug_files:
                            file_path = generator.get_asset_path(proc_sid, file_type)
                            if os.path.exists(file_path):
                                with open(file_path, 'r') as f:
                                    zip_file.writestr(
                                        f"{proc_sid}.{file_type.replace('debug_', 'debug.')}.csv",
                                        f.read()
                                    )
                    
                    st.download_button(
                        label="📦 Debug Files (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"{proc_sid}.debug.zip",
                        mime="application/zip"
                    )
                
                st.divider()
            
            # Summary
            st.subheader("Generation Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Protocols Generated:** {len(st.session_state.generated_protocols)}")
                st.write("**Protocol SIDs:**")
                for sid in st.session_state.generated_protocols:
                    st.write(f"- {sid}")
            
            with col2:
                st.info(f"""
                **Files per Protocol:**
                - OT2 Protocol JSON (main file)
                - Labels PDF
                - Debug Steps CSV
                - Debug Wells CSV
                - Debug Sources CSV
                - Debug Samples CSV
                - Debug Config YAML
                """)
            
            # Clear and restart option
            if st.button("Generate New Batch"):
                for key in ['composition_data', 'batch_config', 'source_config', 
                           'calculated_samples', 'protocol_config', 'generated_protocols', 'generator']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
