import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import math
from pathlib import Path
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试设置更好的中文字体
try:
    # Windows系统常用中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    for font_name in chinese_fonts:
        try:
            font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=[font_name])))
            plt.rcParams['font.family'] = font_name
            break
        except:
            continue
except:
    # 如果找不到中文字体，使用默认字体
    plt.rcParams['font.family'] = 'DejaVu Sans'

# Import functions from zernike_test.py
from zernike_test import (
    zernike_polynomial, get_zernike_index, generate_wavefront_error,
    list_available_measurements
)


def check_chinese_fonts():
    """检查系统可用的中文字体"""
    print("正在检查系统中文字体...")
    chinese_fonts = []
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 常见的中文字体名称
    chinese_font_names = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'KaiTi', 'FangSong',
        'Microsoft JhengHei', 'PingFang SC', 'Heiti SC', 'STHeiti',
        'WenQuanYi Micro Hei', 'Noto Sans CJK', 'Source Han Sans'
    ]
    
    for font_name in chinese_font_names:
        if font_name in all_fonts:
            chinese_fonts.append(font_name)
    
    if chinese_fonts:
        print(f"找到以下中文字体: {chinese_fonts}")
        return chinese_fonts[0]  # 返回第一个找到的中文字体
    else:
        print("未找到系统中文字体，将使用默认字体")
        return None


# 在导入后立即检查并设置最佳中文字体
def setup_chinese_font():
    """设置最佳的中文字体"""
    best_font = check_chinese_fonts()
    if best_font:
        plt.rcParams['font.family'] = [best_font, 'DejaVu Sans']
        print(f"已设置中文字体: {best_font}")
    else:
        print("使用默认字体配置")

# 调用字体设置
setup_chinese_font()


def load_all_measurements_from_json(json_file_path):
    """
    Load all measurements from JSON file for both eyes
    
    Returns:
    dict: Dictionary containing all measurements for each eye
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        measurements = {}
        patient_info = {
            'name': data.get('name', 'Unknown'),
            'age': data.get('age', 'Unknown'),
            'gender': data.get('gender', 'Unknown'),
            'date': data.get('date', 'Unknown')
        }
        
        for eye in ['left', 'right']:
            if eye in data and 'points' in data[eye]:
                eye_measurements = {}
                for point in data[eye]['points']:
                    if point.get('result'):
                        key = (point['offaxis'], point['meridian'])
                        
                        # Calculate median values from all measurements
                        all_spheres = [r['sphere'] for r in point['result']]
                        all_cyls = [r['cyl'] for r in point['result']]
                        all_axes = [r['axis'] for r in point['result']]
                        all_zernikes = [r['zernike'] for r in point['result']]
                        
                        # Use median values or pre-calculated median if available
                        if 'median' in point:
                            median_data = point['median']
                        else:
                            median_data = {
                                'sphere': np.median(all_spheres),
                                'cyl': np.median(all_cyls),
                                'axis': np.median(all_axes),
                                'zernike': np.median(all_zernikes, axis=0).tolist()
                            }
                        
                        eye_measurements[key] = {
                            'sphere': median_data['sphere'],
                            'cyl': median_data['cyl'],
                            'axis': median_data['axis'],
                            'zernike': median_data['zernike'],
                            'offaxis': point['offaxis'],
                            'meridian': point['meridian']
                        }
                
                if eye_measurements:
                    measurements[eye] = eye_measurements
        
        return measurements, patient_info
    
    except Exception as e:
        print(f"Error loading measurements from {json_file_path}: {str(e)}")
        return {}, {}


def create_3d_refractive_topography(measurements, patient_info, save_path=None):
    """
    Create 3D refractive topography map for available eyes
    """
    available_eyes = list(measurements.keys())
    num_eyes = len(available_eyes)
    
    if num_eyes == 0:
        print("No measurement data found")
        return
    
    # Set up the figure for 3D plots
    fig_width = 18 if num_eyes == 2 else 10
    fig = plt.figure(figsize=(fig_width, 8))
    
    # Color map for sphere values
    cmap = plt.cm.RdYlBu_r
    
    for idx, eye in enumerate(available_eyes):
        ax = fig.add_subplot(1, num_eyes, idx + 1, projection='3d')
        eye_data = measurements[eye]
        
        # Prepare data for 3D plotting
        positions = []
        spheres = []
        
        for (offaxis, meridian), data in eye_data.items():
            # Convert polar coordinates to Cartesian
            theta_rad = np.radians(meridian)
            x = offaxis * np.cos(theta_rad)
            y = offaxis * np.sin(theta_rad)
            positions.append((x, y))
            spheres.append(data['sphere'])
        
        if not positions:
            continue
        
        positions = np.array(positions)
        spheres = np.array(spheres)
        
        # Create interpolation grid
        x_min, x_max = positions[:, 0].min() - 5, positions[:, 0].max() + 5
        y_min, y_max = positions[:, 1].min() - 5, positions[:, 1].max() + 5
        
        # Create regular grid
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        X_grid, Y_grid = np.meshgrid(xi, yi)
        
        # Interpolate sphere values onto grid
        Z_grid = griddata(positions, spheres, (X_grid, Y_grid), method='cubic', fill_value=0)
        
        # Apply circular mask to limit the interpolation to reasonable areas
        center_x, center_y = 0, 0
        max_radius = 35
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                distance = np.sqrt((X_grid[i, j] - center_x)**2 + (Y_grid[i, j] - center_y)**2)
                if distance > max_radius:
                    Z_grid[i, j] = np.nan
        
        # Create 3D surface plot
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap, 
                              alpha=0.8, antialiased=True, shade=True,
                              linewidth=0, rcount=50, ccount=50)
        
        # Add contour lines on the bottom (xy plane)
        z_bottom = np.nanmin(Z_grid) - (np.nanmax(Z_grid) - np.nanmin(Z_grid)) * 0.1
        contour_lines = ax.contour(X_grid, Y_grid, Z_grid, 
                                  levels=10, colors='gray', alpha=0.6, 
                                  linestyles='solid', linewidths=0.8)
        
        # Project contours to bottom - handle 3D contour differently
        try:
            # For 3D contours, we need to access the line segments differently
            if hasattr(contour_lines, 'allsegs'):
                for level_segs in contour_lines.allsegs:
                    for seg in level_segs:
                        if len(seg) > 0:
                            ax.plot(seg[:, 0], seg[:, 1], z_bottom, 
                                   color='gray', alpha=0.4, linewidth=0.5)
        except (AttributeError, IndexError):
            # Skip contour projection if not supported in this matplotlib version
            pass
        
        # Add scatter points for actual measurements
        ax.scatter(positions[:, 0], positions[:, 1], spheres, 
                  c=spheres, cmap=cmap, s=120, alpha=1.0, 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Add value annotations near the points
        for pos, sphere in zip(positions, spheres):
            ax.text(pos[0], pos[1], sphere + 0.5, f'{sphere:.2f}D', 
                   fontsize=9, ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('水平离心距 (度)', fontsize=12, labelpad=10)
        ax.set_ylabel('垂直离心距 (度)', fontsize=12, labelpad=10)
        ax.set_zlabel('球镜度 (D)', fontsize=12, labelpad=10)
        ax.set_title(f'{eye.capitalize()} 眼 - 三维屈光地形图', fontsize=14, fontweight='bold', pad=20)
        
        # Enhance the grid appearance
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.5))
        
        # Set viewing angle for better visualization
        ax.view_init(elev=35, azim=45)
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if not np.all(np.isnan(Z_grid)):
            z_min, z_max = np.nanmin(Z_grid), np.nanmax(Z_grid)
            z_range = z_max - z_min
            ax.set_zlim(z_min - z_range * 0.1, z_max + z_range * 0.2)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('球镜度 (D)', rotation=270, labelpad=15)
    
    # Set main title
    main_title = f"三维屈光地形图 - {patient_info['name']} (年龄: {patient_info['age']}, 日期: {patient_info['date']})"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        base_name = os.path.splitext(save_path)[0]
        save_3d_path = f"{base_name}_3d.png"
        plt.savefig(save_3d_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_refractive_topography(measurements, patient_info, save_path=None):
    """
    Create refractive topography map for available eyes
    """
    available_eyes = list(measurements.keys())
    num_eyes = len(available_eyes)
    
    if num_eyes == 0:
        print("No measurement data found")
        return
    
    # Set up the figure with extra width for statistics
    fig_width = 18 if num_eyes == 2 else 12
    fig, axes = plt.subplots(1, num_eyes, figsize=(fig_width, 8))
    if num_eyes == 1:
        axes = [axes]
    
    # Define retinal regions mapping (offaxis, meridian) -> region name
    region_mapping = {
        (0, 0): "Fovea",
        (15, 0): "Nasal 15°",
        (15, 45): "Sup-nasal 15°",
        (15, 90): "Superior 15°",
        (15, 135): "Sup-temporal 15°",
        (15, 180): "Temporal 15°",
        (15, 225): "Inf-temporal 15°",
        (15, 270): "Inferior 15°",
        (15, 315): "Inf-nasal 15°",
        (30, 0): "Nasal 30°",
        (30, 45): "Sup-nasal 30°",
        (30, 90): "Superior 30°",
        (30, 135): "Sup-temporal 30°",
        (30, 180): "Temporal 30°",
        (30, 225): "Inf-temporal 30°",
        (30, 270): "Inferior 30°",
        (30, 315): "Inf-nasal 30°"
    }
    
    # Color map for sphere values
    cmap = plt.cm.RdYlBu_r
    
    for idx, eye in enumerate(available_eyes):
        ax = axes[idx]
        eye_data = measurements[eye]
        
        # Prepare data for plotting
        positions = []
        spheres = []
        
        for (offaxis, meridian), data in eye_data.items():
            # Convert polar coordinates to Cartesian for plotting
            theta_rad = np.radians(meridian)
            x = offaxis * np.cos(theta_rad)
            y = offaxis * np.sin(theta_rad)
            positions.append((x, y))
            spheres.append(data['sphere'])
        
        if not positions:
            continue
            
        # Create scatter plot
        positions = np.array(positions)
        spheres = np.array(spheres)
        
        # Determine color range
        vmin, vmax = np.min(spheres), np.max(spheres)
        v_range = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_range, v_range
        
        # Statistical analysis
        mean_sphere = np.mean(spheres)
        std_sphere = np.std(spheres, ddof=1)  # Sample standard deviation
        var_sphere = np.var(spheres, ddof=1)  # Sample variance
        
        # Detect outliers using modified Z-score (more robust than standard Z-score)
        median_sphere = np.median(spheres)
        mad = np.median(np.abs(spheres - median_sphere))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (spheres - median_sphere) / mad if mad > 0 else np.zeros_like(spheres)
        
        # Outliers: |modified Z-score| > 3.5
        outlier_threshold = 3.5
        outliers_mask = np.abs(modified_z_scores) > outlier_threshold
        outliers = spheres[outliers_mask]
        outlier_positions = positions[outliers_mask]
        
        # Create the scatter plot
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=spheres, cmap=cmap, s=200, 
                           vmin=vmin, vmax=vmax, alpha=0.8)
        
        # Highlight outliers with red circles
        if len(outliers) > 0:
            ax.scatter(outlier_positions[:, 0], outlier_positions[:, 1], 
                      s=250, facecolors='none', edgecolors='red', 
                      linewidths=3, label='异常点')
        
        # Add text annotations for sphere values
        for pos, sphere in zip(positions, spheres):
            ax.annotate(f'{sphere:.2f}D', 
                       (pos[0], pos[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        # Set up the plot - expand limits to include all data points
        # Find the actual range of data points
        max_radius = max(pos[0] for pos in eye_data.keys())
        axis_limit = max_radius + 5  # Add some padding
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('水平离心距 (度)')
        ax.set_ylabel('垂直离心距 (度)')
        ax.set_title(f'{eye.capitalize()} 眼 - 屈光地形图')
        
        # Add concentric circles for reference - dynamic based on data
        unique_offaxis = sorted(set(pos[0] for pos in eye_data.keys()))
        reference_radii = [r for r in unique_offaxis if r > 0]  # Use actual offaxis distances
        for radius in reference_radii:
            circle = plt.Circle((0, 0), radius, fill=False, 
                              color='gray', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        # Add statistical information text on the right side
        stats_text = f"统计分析 ({eye.capitalize()} 眼):\n\n"
        stats_text += f"样本数量: {len(spheres)}\n"
        stats_text += f"平均值: {mean_sphere:.3f} D\n"
        stats_text += f"标准差: {std_sphere:.3f} D\n"
        stats_text += f"方差: {var_sphere:.3f} D²\n"
        stats_text += f"中位数: {median_sphere:.3f} D\n"
        stats_text += f"范围: [{np.min(spheres):.3f}, {np.max(spheres):.3f}] D\n\n"
        
        if len(outliers) > 0:
            stats_text += f"异常点检测 (修正Z分数法):\n"
            stats_text += f"异常点数量: {len(outliers)}\n"
            for i, (outlier_val, outlier_pos) in enumerate(zip(outliers, outlier_positions)):
                # Find the offaxis and meridian for this outlier
                pos_idx = np.where(np.all(positions == outlier_pos, axis=1))[0][0]
                offaxis_meridian_pairs = list(eye_data.keys())
                offaxis, meridian = offaxis_meridian_pairs[pos_idx]
                stats_text += f"  异常点 {i+1}: {outlier_val:.3f} D\n"
                stats_text += f"    位置: ({offaxis}°, {meridian}°)\n"
        else:
            stats_text += "异常点检测: 无异常点\n"
        
        # Add text box with statistics
        ax.text(1.15, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightgray', alpha=0.8))
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.35)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('球镜度 (D)', rotation=270, labelpad=15)
    
    # Set main title
    main_title = f"屈光地形图 - {patient_info['name']} (年龄: {patient_info['age']}, 日期: {patient_info['date']})"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # Make room for statistics text
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_wavefront_array(measurements, patient_info, save_path=None):
    """
    Create wavefront error maps array for all measurement positions
    Arranged in radial pattern with center position (0°) in the middle
    """
    available_eyes = list(measurements.keys())
    
    for eye in available_eyes:
        eye_data = measurements[eye]
        
        # Group positions by offaxis distance
        offaxis_groups = {}
        for pos in eye_data.keys():
            offaxis = pos[0]
            if offaxis not in offaxis_groups:
                offaxis_groups[offaxis] = []
            offaxis_groups[offaxis].append(pos)
        
        # Calculate all wavefront data first for consistent color scaling
        # Skip first 3 Zernike coefficients (piston, tip, tilt)
        all_wavefronts = []
        wavefront_data = {}
        for pos in eye_data.keys():
            zernike_coeffs = np.array(eye_data[pos]['zernike'])
            # Use only coefficients from index 3 onwards (skip Z0, Z1, Z2)
            if len(zernike_coeffs) > 3:
                modified_coeffs = np.zeros_like(zernike_coeffs)
                modified_coeffs[3:] = zernike_coeffs[3:]
            else:
                modified_coeffs = zernike_coeffs
            X, Y, wf = generate_wavefront_error(modified_coeffs, grid_size=128)
            wavefront_data[pos] = (X, Y, wf)
            all_wavefronts.extend(wf[~np.isnan(wf)].flatten())
        
        if not all_wavefronts:
            continue
        
        # Use percentile-based scaling for better color distribution
        vmin = np.percentile(all_wavefronts, 5)  # 5th percentile
        vmax = np.percentile(all_wavefronts, 95)  # 95th percentile
        
        # Ensure symmetric range around zero for better rainbow visualization
        v_abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_abs_max, v_abs_max
        
        # Create figure with larger size to accommodate radial layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Dynamic axis limits based on data range
        max_radius = max(pos[0] for pos in eye_data.keys()) if eye_data else 50
        # Scale for radial layout (add margin for labels and wavefront maps)
        axis_scale = max_radius / 50 * 6 + 2  # Scale based on max radius
        ax.set_xlim(-axis_scale, axis_scale)
        ax.set_ylim(-axis_scale, axis_scale)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Get unique offaxis distances and meridians, then create compact layout
        unique_offaxis = sorted(set(pos[0] for pos in eye_data.keys()))
        
        def get_compact_radial_positions():
            """Create compact radial layout without gaps for missing data"""
            positions = {}
            
            # Center position
            for pos in eye_data.keys():
                if pos[0] == 0:
                    positions[pos] = (0, 0)
            
            # For each offaxis distance, arrange available meridians compactly
            for offaxis in unique_offaxis:
                if offaxis == 0:
                    continue
                    
                # Get available meridians for this offaxis
                available_meridians = sorted([pos[1] for pos in eye_data.keys() if pos[0] == offaxis])
                
                # Calculate radius based on offaxis distance (dynamic scaling)
                # Scale radius proportionally to offaxis distance
                radius_scale = axis_scale / max_radius if max_radius > 0 else 0.1
                radius = offaxis * radius_scale * 0.8  # 0.8 factor to leave margin
                
                # Distribute available positions evenly around the circle
                num_positions = len(available_meridians)
                for i, meridian in enumerate(available_meridians):
                    # Evenly space available positions, starting from meridian 0°
                    if num_positions == 1:
                        angle = np.radians(meridian)
                    else:
                        # Find the actual angle for this meridian and adjust for compact spacing
                        angle = np.radians(meridian)
                    
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    positions[(offaxis, meridian)] = (x, y)
            
            return positions
        
        # Get all positions
        radial_positions = get_compact_radial_positions()
        
        # Plot each wavefront map
        wavefront_size = 0.8  # Reduced size (half of original)
        
        for pos in eye_data.keys():
            offaxis_val, meridian_val = pos
            X, Y, wavefront = wavefront_data[pos]
            
            # Get position in radial layout
            center_x, center_y = radial_positions[pos]
            
            # Create a new axes for this wavefront map
            ax_pos = [
                (center_x + 7) / 14 - wavefront_size/28,  # Convert to figure coordinates
                (center_y + 6) / 12 - wavefront_size/24,
                wavefront_size/14,
                wavefront_size/12
            ]
            
            wf_ax = fig.add_axes(ax_pos)
            
            # Plot the wavefront with rainbow colormap (jet is more vibrant than rainbow)
            im = wf_ax.contourf(X, Y, wavefront, levels=50, cmap='jet',
                               vmin=vmin, vmax=vmax)
            wf_ax.set_aspect('equal')
            
            # Add circular boundary
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1.5)
            wf_ax.add_patch(circle)
            
            # Remove ticks
            wf_ax.set_xticks([])
            wf_ax.set_yticks([])
            
            # Add label below each wavefront map
            if offaxis_val == 0:
                label = "0°"
                label_y = center_y - 1.0  # Closer spacing
            else:
                label = f"{offaxis_val}°"
                label_y = center_y - 1.0  # Closer spacing
            
            ax.text(center_x, label_y, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Add region labels - dynamic positioning based on axis scale
        label_distance = axis_scale * 0.85
        ax.text(0, label_distance, '上方视网膜', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -label_distance, '下方视网膜', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(-label_distance, 0, '颞侧视网膜', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
        ax.text(label_distance, 0, '鼻侧视网膜', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
        
        # Add title
        main_title = f"波前误差图阵列 - {eye.capitalize()} 眼\n{patient_info['name']} (年龄: {patient_info['age']}, 日期: {patient_info['date']})"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.02, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('波前误差 (μm)', rotation=270, labelpad=15)
        
        if save_path:
            base_name = os.path.splitext(save_path)[0]
            eye_save_path = f"{base_name}_{eye}_wavefront_radial.png"
            plt.savefig(eye_save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def process_all_json_files(data_folder, output_folder=None):
    """
    Process all JSON files in the specified folder
    """
    data_path = Path(data_folder)
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = None
    
    json_files = list(data_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            # Load measurements
            measurements, patient_info = load_all_measurements_from_json(str(json_file))
            
            if not measurements:
                print(f"No valid measurements found in {json_file.name}")
                continue
            
            print(f"Found data for: {list(measurements.keys())}")
            
            # Create refractive topography (2D)
            refractive_save_path = None
            if output_path:
                refractive_save_path = output_path / f"{json_file.stem}_refractive_topography.png"
            
            create_refractive_topography(measurements, patient_info, refractive_save_path)
            
            # Create 3D refractive topography
            refractive_3d_save_path = None
            if output_path:
                refractive_3d_save_path = output_path / f"{json_file.stem}_refractive_3d_topography.png"
            
            create_3d_refractive_topography(measurements, patient_info, refractive_3d_save_path)
            
            # Create wavefront arrays
            wavefront_save_path = None
            if output_path:
                wavefront_save_path = output_path / f"{json_file.stem}_wavefront_array.png"
            
            create_wavefront_array(measurements, patient_info, wavefront_save_path)
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")


def analyze_single_file(json_file_path, output_folder=None):
    """
    Analyze a single JSON file
    """
    print(f"Analyzing: {json_file_path}")
    
    # Load measurements
    measurements, patient_info = load_all_measurements_from_json(json_file_path)
    
    if not measurements:
        print("No valid measurements found")
        return
    
    print(f"Found data for: {list(measurements.keys())}")
    
    # Create save paths if output folder is specified
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        file_stem = Path(json_file_path).stem
        
        refractive_save_path = output_path / f"{file_stem}_refractive_topography.png"
        refractive_3d_save_path = output_path / f"{file_stem}_refractive_3d_topography.png"
        wavefront_save_path = output_path / f"{file_stem}_wavefront_array.png"
    else:
        refractive_save_path = None
        refractive_3d_save_path = None
        wavefront_save_path = None
    
    # Create visualizations
    create_refractive_topography(measurements, patient_info, refractive_save_path)
    create_3d_refractive_topography(measurements, patient_info, refractive_3d_save_path)
    create_wavefront_array(measurements, patient_info, wavefront_save_path)


if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = "testdata_aier"
    OUTPUT_FOLDER = "output_analysis"  # Set to None if you don't want to save files
    
    # Option 1: Process all JSON files in the folder
    # process_all_json_files(DATA_FOLDER, OUTPUT_FOLDER)
    
    # Option 2: Process a single file (example with 095.json)
    single_file_path = "testdata_aier/033.json"
    if os.path.exists(single_file_path):
        analyze_single_file(single_file_path, OUTPUT_FOLDER)
    else:
        print(f"File {single_file_path} not found")
    
    # Option 3: Process specific files
    # specific_files = ["testdata_aier/001.json", "testdata_aier/095.json"]
    # for file_path in specific_files:
    #     if os.path.exists(file_path):
    #         analyze_single_file(file_path, OUTPUT_FOLDER)