#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

class DogmaMapperNode : public rclcpp::Node {
public:
  DogmaMapperNode() : Node("dogma_mapper_node") {
    // 1. Grid Parameters (20m x 20m studio at 0.1m per cell = 200x200 cells)
    resolution_ = 0.1;
    width_ = 200;
    height_ = 200;
    origin_x_ = -10.0;
    origin_y_ = -10.0;

    // Initialize log-odds grid to 0.0 (50% unknown probability)
    log_odds_grid_.resize(width_ * height_, 0.0f);

    // 2. TF2 Buffer and Listener for Camera-to-World Transformations
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 3. Subscribers
    tracks_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
      "/tracked_drones", 10,
      std::bind(&DogmaMapperNode::tracks_callback, this, std::placeholders::_1));

    // 4. Map Publisher (Subscribable by Foxglove Studio over WiFi)
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/studio_map", 10);

    // 5. Periodic Timer (10 Hz) to decay dynamic cells and publish map
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&DogmaMapperNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "DOGMA Mapper Node initialized (20m x 20m @ 0.1m res).");
  }

private:
  void tracks_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
    for (const auto & marker : msg->markers) {
      // Only process the drone position spheres, skip velocity arrows
      if (marker.ns != "tracked_drones") {
        continue;
      }

      geometry_msgs::msg::PointStamped cam_pt, map_pt;
      cam_pt.header = marker.header;
      cam_pt.point = marker.pose.position;

      try {
        // Transform detection from camera frame into global map frame
        map_pt = tf_buffer_->transform(cam_pt, "map", tf2::durationFromSec(0.05));
        mark_occupied(map_pt.point.x, map_pt.point.y, 0.35);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Waiting for map -> camera transform: %s", ex.what());
      }
    }
  }

  void mark_occupied(double wx, double wy, double radius) {
    int cell_radius = static_cast<int>(std::ceil(radius / resolution_));
    int cx = static_cast<int>((wx - origin_x_) / resolution_);
    int cy = static_cast<int>((wy - origin_y_) / resolution_);

    for (int dx = -cell_radius; dx <= cell_radius; ++dx) {
      for (int dy = -cell_radius; dy <= cell_radius; ++dy) {
        if (dx * dx + dy * dy > cell_radius * cell_radius) continue;
        int gx = cx + dx;
        int gy = cy + dy;

        if (gx >= 0 && gx < width_ && gy >= 0 && gy < height_) {
          int idx = gy * width_ + gx;
          // Increase log-odds (clamped to max certainty)
          log_odds_grid_[idx] = std::min(5.0f, log_odds_grid_[idx] + 1.5f);
        }
      }
    }
  }

  void timer_callback() {
    auto grid_msg = nav_msgs::msg::OccupancyGrid();
    grid_msg.header.stamp = this->now();
    grid_msg.header.frame_id = "map";

    grid_msg.info.resolution = resolution_;
    grid_msg.info.width = width_;
    grid_msg.info.height = height_;
    grid_msg.info.origin.position.x = origin_x_;
    grid_msg.info.origin.position.y = origin_y_;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(width_ * height_);

    for (size_t i = 0; i < log_odds_grid_.size(); ++i) {
      // Apply dynamic decay so moving targets clear out over time
      log_odds_grid_[i] *= 0.90f;
      if (std::abs(log_odds_grid_[i]) < 0.05f) {
        log_odds_grid_[i] = 0.0f;
      }

      // Convert log-odds to ROS OccupancyGrid probability [0 to 100]
      float prob = 1.0f - (1.0f / (1.0f + std::exp(log_odds_grid_[i])));
      grid_msg.data[i] = static_cast<int8_t>(prob * 100.0f);
    }

    map_pub_->publish(grid_msg);
  }

  double resolution_;
  int width_;
  int height_;
  double origin_x_;
  double origin_y_;
  std::vector<float> log_odds_grid_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr tracks_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DogmaMapperNode>());
  rclcpp::shutdown();
  return 0;
}

