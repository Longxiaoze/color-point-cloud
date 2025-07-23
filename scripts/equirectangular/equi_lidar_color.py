import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.point_cloud2 import read_points, create_cloud_xyzrgb
from cv_bridge import CvBridge
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

class PanoramicPointCloudColorizer(Node):
    def __init__(self):
        super().__init__('panoramic_pointcloud_colorizer')

        # 参数声明
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('pointcloud_topic', '/pointcloud')
        
        # 获取参数
        self.calibration_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value

        # 加载外参
        self.load_calibration()
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 图像和点云缓存
        self.latest_image = None
        self.latest_cloud = None

        # 创建订阅者
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
            
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pointcloud_callback, 10)

        # 创建发布者
        self.color_pub = self.create_publisher(PointCloud2, 'colorized_pointcloud', 10)

    def load_calibration(self):
        """从YAML文件加载LiDAR到相机的外参"""
        with open(self.calibration_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        # 解析平移和旋转参数
        trans = calib_data['transform']['translation']
        rot = calib_data['transform']['rotation']
        
        # 构造变换矩阵
        self.T_lidar_to_camera = np.eye(4)
        self.T_lidar_to_camera[:3, 3] = [trans['x'], trans['y'], trans['z']]
        self.T_lidar_to_camera[:3, :3] = R.from_quat([
            rot['x'], rot['y'], rot['z'], rot['w']
        ]).as_matrix()

    def image_callback(self, msg):
        """图像回调函数"""
        self.latest_image = msg

    def pointcloud_callback(self, cloud_msg):
        """点云回调函数"""
        if self.latest_image is None:
            return

        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
            img_height, img_width = cv_image.shape[:2]
            
            # 处理点云
            colored_points = []
            for point in read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point[0], point[1], point[2]
                
                # 转换到相机坐标系
                point_cam = self.T_lidar_to_camera @ np.array([x, y, z, 1.0])
                xc, yc, zc = point_cam[:3]
                
                # 过滤相机后方的点
                if xc <= 0:
                    continue
                
                # 计算球面坐标
                r = np.sqrt(xc**2 + yc**2)
                if r < 1e-6:
                    continue
                    
                theta = np.arctan2(yc, xc)  # 方位角
                phi = np.arctan2(zc, r)     # 仰角
                
                # 映射到图像坐标
                u = (theta + np.pi) / (2 * np.pi) * img_width
                v = (phi + np.pi/2) / np.pi * img_height
                
                u_int, v_int = int(u), int(v)
                
                # 检查坐标有效性
                if 0 <= u_int < img_width and 0 <= v_int < img_height:
                    b, g, r = cv_image[v_int, u_int]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                    colored_points.append((x, y, z, rgb))
            
            # 发布带颜色的点云
            if colored_points:
                cloud_out = create_cloud_xyzrgb(cloud_msg.header, colored_points)
                self.color_pub.publish(cloud_out)
                
        except Exception as e:
            self.get_logger().error(f'处理数据时出错: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PanoramicPointCloudColorizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()