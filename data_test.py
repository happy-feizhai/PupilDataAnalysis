import socket
import struct
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.special import factorial
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


class ThorlabsZ80Analyzer:
    """Thorlabs Z80标准泽尼克分析器"""

    # Thorlabs Z80 泽尼克索引映射
    ZERNIKE_MAP = {
        0: "Piston",
        1: "Tip y",
        2: "Tilt x",
        3: "Astigmatism ±45°",
        4: "Defocus",
        5: "Astigmatism 0/90°",
        6: "Trefoil y",
        7: "Coma y",
        8: "Coma x",
        9: "Trefoil x",
        10: "Spherical"
    }

    @staticmethod
    def calculate_optical_params_z80(zernike_coeffs, beam_diameter_mm):
        """根据Thorlabs Z80标准计算光学参数

        Thorlabs Z80索引：
        - 索引4: Defocus (球镜)
        - 索引3: Astigmatism ±45°
        - 索引5: Astigmatism 0/90°
        """

        # 获取关键系数
        Z_defocus = zernike_coeffs[4] if len(zernike_coeffs) > 4 else 0  # Defocus
        Z_astig45 = zernike_coeffs[3] if len(zernike_coeffs) > 3 else 0  # Astigmatism ±45°
        Z_astig90 = zernike_coeffs[5] if len(zernike_coeffs) > 5 else 0  # Astigmatism 0/90°

        # 瞳孔半径
        pupil_radius = beam_diameter_mm / 2.0

        # 计算球镜度
        # 标准公式: Sphere = -4√3 * Z_defocus / r²
        sphere = -4 * np.sqrt(3) * Z_defocus / (pupil_radius ** 2)

        # 计算柱镜度和轴向
        # 注意：Thorlabs的散光排序与标准不同
        # Z3是±45°散光，Z5是0/90°散光
        astig_magnitude = np.sqrt(Z_astig45 ** 2 + Z_astig90 ** 2)
        cylinder = -2 * np.sqrt(6) * astig_magnitude / (pupil_radius ** 2)

        # 轴向计算需要考虑Thorlabs的特殊排序
        # Z3 (±45°) 和 Z5 (0/90°) 的关系
        if Z_astig45 != 0 or Z_astig90 != 0:
            # 注意：轴向定义可能需要调整
            axis = np.degrees(0.5 * np.arctan2(-Z_astig45, -Z_astig90))
            if axis < 0:
                axis += 180
        else:
            axis = 0

        # 傅里叶系数
        M = sphere - cylinder / 2
        J0 = -cylinder / 2 * np.cos(2 * np.radians(axis))
        J45 = -cylinder / 2 * np.sin(2 * np.radians(axis))

        return sphere, cylinder, axis, M, J0, J45

    @staticmethod
    def generate_zernike_z80(idx, size=100):
        """生成Thorlabs Z80标准的泽尼克多项式"""

        # 创建归一化坐标
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)

        # 圆形掩膜
        mask = rho <= 1
        Z = np.zeros((size, size))

        # 根据Thorlabs Z80定义生成多项式
        if idx == 0:  # Piston
            Z[mask] = 1
        elif idx == 1:  # Tip y
            Z[mask] = 2 * rho[mask] * np.sin(theta[mask])
        elif idx == 2:  # Tilt x
            Z[mask] = 2 * rho[mask] * np.cos(theta[mask])
        elif idx == 3:  # Astigmatism ±45°
            Z[mask] = np.sqrt(6) * rho[mask] ** 2 * np.sin(2 * theta[mask])
        elif idx == 4:  # Defocus
            Z[mask] = np.sqrt(3) * (2 * rho[mask] ** 2 - 1)
        elif idx == 5:  # Astigmatism 0/90°
            Z[mask] = np.sqrt(6) * rho[mask] ** 2 * np.cos(2 * theta[mask])
        elif idx == 6:  # Trefoil y
            Z[mask] = np.sqrt(8) * rho[mask] ** 3 * np.sin(3 * theta[mask])
        elif idx == 7:  # Coma y
            Z[mask] = np.sqrt(8) * (3 * rho[mask] ** 3 - 2 * rho[mask]) * np.sin(theta[mask])
        elif idx == 8:  # Coma x
            Z[mask] = np.sqrt(8) * (3 * rho[mask] ** 3 - 2 * rho[mask]) * np.cos(theta[mask])
        elif idx == 9:  # Trefoil x
            Z[mask] = np.sqrt(8) * rho[mask] ** 3 * np.cos(3 * theta[mask])
        elif idx == 10:  # Spherical
            Z[mask] = np.sqrt(5) * (6 * rho[mask] ** 4 - 6 * rho[mask] ** 2 + 1)

        Z[~mask] = np.nan
        return Z

    @staticmethod
    def reconstruct_wavefront_z80(zernike_coeffs, size=100, exclude_piston_tilt=True):
        """使用Thorlabs Z80标准重建波前"""

        wavefront = np.zeros((size, size))

        # 决定起始索引
        start_idx = 3 if exclude_piston_tilt else 0

        # 累加泽尼克项
        analyzer = ThorlabsZ80Analyzer()
        for idx in range(start_idx, min(len(zernike_coeffs), 11)):
            Z = analyzer.generate_zernike_z80(idx, size)
            wavefront += zernike_coeffs[idx] * Z

        return wavefront

    @staticmethod
    def calculate_pv_rms(wavefront):
        """计算波前的PV和RMS值"""
        valid_data = wavefront[~np.isnan(wavefront)]
        if len(valid_data) == 0:
            return 0, 0

        pv = np.max(valid_data) - np.min(valid_data)
        rms = np.sqrt(np.mean(valid_data ** 2))

        return pv, rms





class WFSDataReceiver:
    """波前传感器数据接收器"""

    # 数据包头格式：魔数(I) + JSON长度(I) + 波前数据长度(I) + 行数(H) + 列数(H)
    HEADER_FORMAT = 'IIIHH'  # I=unsigned int, H=unsigned short
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAGIC_NUMBER = 0x57465330  # 'WFS0'

    def __init__(self, host: str = 'localhost', port: int = 6000):
        self.host = host
        self.port = port
        self.socket = None
        self.analyzer = ThorlabsZ80Analyzer()

    def connect(self) -> bool:
        """连接到TCP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None

    def _recv_exact(self, size: int) -> bytes:
        """精确接收指定字节数的数据"""
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def receive_data(self) -> Optional[Dict[str, Any]]:
        """接收一个完整的数据包"""
        if not self.socket:
            return None

        try:
            # 1. 接收包头
            header_data = self._recv_exact(self.HEADER_SIZE)
            magic, json_len, wf_len, rows, cols = struct.unpack(
                self.HEADER_FORMAT, header_data
            )

            # 2. 验证魔数
            if magic != self.MAGIC_NUMBER:
                # 可能是旧格式的纯JSON，尝试按行读取
                return self._receive_legacy_json(header_data)

            # 3. 接收JSON数据
            json_data = self._recv_exact(json_len).decode('utf-8')
            result = json.loads(json_data.strip())

            # 4. 接收波前数据
            if wf_len > 0:
                wavefront_data = self._recv_exact(wf_len)

                # 5. 将二进制数据转换为numpy数组
                wavefront_flat = np.frombuffer(wavefront_data, dtype=np.float32)
                wavefront_2d = wavefront_flat.reshape((rows, cols))

                # 添加到结果字典
                result['wavefront_array'] = wavefront_2d
                result['wavefront_shape'] = (rows, cols)

            return result

        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def _receive_legacy_json(self, initial_data: bytes) -> Optional[Dict[str, Any]]:
        """处理旧格式的纯JSON数据（向后兼容）"""
        try:
            # 读取直到换行符
            line = initial_data
            while b'\n' not in line:
                chunk = self.socket.recv(1)
                if not chunk:
                    break
                line += chunk

            json_str = line.decode('utf-8').strip()
            return json.loads(json_str)
        except:
            return None

    def analyze_and_compare(self, data: Dict[str, Any]):
        """使用Thorlabs Z80标准分析"""

        if 'zernike' not in data:
            print("No Zernike data available")
            return

        zernike_coeffs = data['zernike']
        analyzer = ThorlabsZ80Analyzer()

        # 获取光束直径
        beam_diameter = (data.get('diameter_x', 8) + data.get('diameter_y', 8)) / 2

        print("\n=== Thorlabs Z80 泽尼克分析 ===")
        print(f"光束直径: {beam_diameter:.3f} mm")

        # 显示关键系数
        print("\n关键泽尼克系数:")
        for idx in [0, 1, 2, 3, 4, 5]:
            if idx < len(zernike_coeffs):
                print(f"  Z{idx} ({analyzer.ZERNIKE_MAP.get(idx, '')}): {zernike_coeffs[idx]:.5f}")

        # 计算光学参数
        sphere, cyl, axis, M, J0, J45 = analyzer.calculate_optical_params_z80(
            zernike_coeffs, beam_diameter
        )

        print("\n--- 光学参数比较 ---")
        print(f"球镜度 (Sphere):")
        print(f"  接收: {data.get('sphere', 0):.3f} D")
        print(f"  计算: {sphere:.3f} D")
        print(f"  差异: {abs(data.get('sphere', 0) - sphere):.3f} D")

        print(f"柱镜度 (Cylinder):")
        print(f"  接收: {data.get('cyl', 0):.3f} D")
        print(f"  计算: {cyl:.3f} D")
        print(f"  差异: {abs(data.get('cyl', 0) - cyl):.3f} D")

        print(f"轴向 (Axis):")
        print(f"  接收: {data.get('axis', 0):.1f}°")
        print(f"  计算: {axis:.1f}°")

        # 重建波前
        reconstructed_wf = analyzer.reconstruct_wavefront_z80(
            zernike_coeffs, size=100, exclude_piston_tilt=True
        )

        # 计算PV和RMS
        valid_data = reconstructed_wf[~np.isnan(reconstructed_wf)]
        calc_pv = np.max(valid_data) - np.min(valid_data)
        calc_rms = np.sqrt(np.mean(valid_data ** 2))

        print("\n--- 波前参数比较 ---")
        print(f"PV值:")
        print(f"  接收: {data.get('wf_pv', 0):.3f} µm")
        print(f"  计算: {calc_pv:.3f} µm")

        print(f"RMS值:")
        print(f"  接收: {data.get('wf_rms', 0):.3f} µm")
        print(f"  计算: {calc_rms:.3f} µm")

        # 可视化
        self.visualize_wavefront_comparison(reconstructed_wf, data)

    def visualize_wavefront_comparison(self, reconstructed_wf, data):
        """改进的可视化"""

        fig, axes = plt.subplots(1, 3 if 'wavefront_array' in data else 2,
                                 figsize=(15 if 'wavefront_array' in data else 10, 5))

        # 泽尼克重建波前
        im1 = axes[0].imshow(reconstructed_wf, cmap='RdBu_r', interpolation='bilinear')
        axes[0].set_title('泽尼克重建波前\n(Thorlabs Z80)')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0], label='波前 (µm)')

        if 'wavefront_array' in data:
            # 原始测量波前
            original_wf = data['wavefront_array']
            im2 = axes[1].imshow(original_wf, cmap='RdBu_r', interpolation='nearest')
            axes[1].set_title('原始测量波前')
            axes[1].set_xlabel('X (pixels)')
            axes[1].set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=axes[1], label='波前 (µm)')

            # 3D视图
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            x = np.arange(reconstructed_wf.shape[1])
            y = np.arange(reconstructed_wf.shape[0])
            X, Y = np.meshgrid(x, y)
            mask = ~np.isnan(reconstructed_wf)
            # The 'mask' variable is no longer needed for this part
            ax3.plot_surface(X, Y, reconstructed_wf,  # Pass the original 2D arrays
                             cmap='RdBu_r', alpha=0.8)
            ax3.set_title('重建波前3D')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('波前 (µm)')

        plt.tight_layout()
        plt.show()

    def receive_continuous(self, callback=None):
        """连续接收数据"""
        while True:
            data = self.receive_data()
            if data is None:
                break

            if callback:
                callback(data)
            else:
                self.print_data(data)

    @staticmethod
    def print_data(data: Dict[str, Any]):
        """打印接收到的数据"""
        print("\n--- WFS Data Received ---")

        # 打印基本参数
        for key in ['centroid_x', 'centroid_y', 'diameter_x', 'diameter_y',
                    'wf_pv', 'wf_rms', 'wf_wrms', 'sphere', 'cyl', 'axis', 'roc']:
            if key in data:
                print(f"{key:15s}: {data[key]:8.3f}")

        # 打印泽尼克系数
        if 'zernike' in data:
            print(f"Zernike coeffs : {len(data['zernike'])} values")
            print(f"  First 5: {data['zernike'][:5]}")

        # 打印波前数据信息
        if 'wavefront_array' in data:
            wf = data['wavefront_array']
            print(f"Wavefront array: {wf.shape}")
            print(f"  Min: {np.nanmin(wf):.3f}, Max: {np.nanmax(wf):.3f}")
            print(f"  Mean: {np.nanmean(wf):.3f}, Std: {np.nanstd(wf):.3f}")
            print(f"  Valid points: {np.sum(~np.isnan(wf))}")


# 使用示例
def main():
    # 创建接收器
    receiver = WFSDataReceiver('localhost', 6000)  # 本地连接

    if not receiver.connect():
        return

    try:
        # 接收单个数据包
        data = receiver.receive_data()
        if data:
            receiver.print_data(data)

            # 执行泽尼克分析和比较
            receiver.analyze_and_compare(data)

    finally:
        receiver.disconnect()


if __name__ == "__main__":
    main()