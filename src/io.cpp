#include <plain_slam/io.hpp>

#include <cstring>
#include <sstream>
#include <string>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>

namespace pslam {

namespace {

bool ParseBinaryCompressedPCD(
  const std::string& fname,
  PointCloud3f& cloud,
  std::vector<float>& intensities) {
  pcl::PCLPointCloud2 pcl_cloud;
  if (pcl::io::loadPCDFile(fname, pcl_cloud) < 0) {
    std::cerr << "[ERROR] Failed to load binary_compressed PCD with PCL: " << fname << std::endl;
    return false;
  }

  const pcl::PCLPointField* x_field = nullptr;
  const pcl::PCLPointField* y_field = nullptr;
  const pcl::PCLPointField* z_field = nullptr;
  const pcl::PCLPointField* intensity_field = nullptr;
  for (const auto& field : pcl_cloud.fields) {
    if (field.name == "x") {
      x_field = &field;
    } else if (field.name == "y") {
      y_field = &field;
    } else if (field.name == "z") {
      z_field = &field;
    } else if (field.name == "intensity") {
      intensity_field = &field;
    }
  }

  if (x_field == nullptr || y_field == nullptr || z_field == nullptr) {
    std::cerr << "[ERROR] binary_compressed PCD must contain x, y, and z fields." << std::endl;
    return false;
  }

  const auto is_supported_float_field = [](const pcl::PCLPointField* field) {
    return field != nullptr &&
           field->datatype == pcl::PCLPointField::FLOAT32 &&
           field->count == 1;
  };

  if (!is_supported_float_field(x_field) ||
      !is_supported_float_field(y_field) ||
      !is_supported_float_field(z_field)) {
    std::cerr << "[ERROR] binary_compressed PCD supports float32 x/y/z fields only." << std::endl;
    return false;
  }

  const bool has_intensity = intensity_field != nullptr;
  if (has_intensity && !is_supported_float_field(intensity_field)) {
    std::cerr << "[ERROR] binary_compressed PCD supports float32 intensity only." << std::endl;
    return false;
  }

  const size_t num_points = static_cast<size_t>(pcl_cloud.width) * static_cast<size_t>(pcl_cloud.height);
  cloud.clear();
  intensities.clear();
  cloud.reserve(num_points);
  if (has_intensity) {
    intensities.reserve(num_points);
  }

  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t* point_ptr = pcl_cloud.data.data() + i * pcl_cloud.point_step;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::memcpy(&x, point_ptr + x_field->offset, sizeof(float));
    std::memcpy(&y, point_ptr + y_field->offset, sizeof(float));
    std::memcpy(&z, point_ptr + z_field->offset, sizeof(float));
    cloud.emplace_back(x, y, z);

    if (has_intensity) {
      float intensity = 0.0f;
      std::memcpy(&intensity, point_ptr + intensity_field->offset, sizeof(float));
      intensities.push_back(intensity);
    }
  }

  return true;
}

} // namespace

bool WritePointCloud(
  const std::string& fname,
  const PointCloud3f& cloud) {
  std::ofstream ofs(fname, std::ios::binary);
  if (!ofs) {
    std::cerr << "[ERROR] Cannot open file for writing: " << fname << std::endl;
    return false;
  }

  uint32_t num_points = static_cast<uint32_t>(cloud.size());
  ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));

  for (size_t i = 0; i < cloud.size(); ++i) {
    const Eigen::Vector3f& p = cloud[i];
    const float data[3] = {p.x(), p.y(), p.z()};
    ofs.write(reinterpret_cast<const char*>(data), sizeof(float) * 3);
  }

  ofs.close();

  return true;
}

bool WritePointCloud(
  const std::string& fname,
  const PointCloud3f& cloud,
  const std::vector<float>& intensities) {
  if (cloud.size() != intensities.size()) {
    std::cerr << "[ERROR] cloud.size() != intensities.size()" << std::endl;
    return false;
  }

  std::ofstream ofs(fname, std::ios::binary);
  if (!ofs) {
    std::cerr << "[ERROR] Cannot open file for writing: " << fname << std::endl;
    return false;
  }

  uint32_t num_points = static_cast<uint32_t>(cloud.size());
  ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));

  for (size_t i = 0; i < cloud.size(); ++i) {
    const Eigen::Vector3f& p = cloud[i];
    const float data[4] = {p.x(), p.y(), p.z(), intensities[i]};
    ofs.write(reinterpret_cast<const char*>(data), sizeof(float) * 4);
  }

  ofs.close();

  return true;
}

bool ReadPointCloud(
  const std::string& fname,
  PointCloud3f& cloud) {
  std::ifstream ifs(fname, std::ios::binary);
  if (!ifs) {
    std::cerr << "[ERROR] Cannot open file for reading: " << fname << std::endl;
    return false;
  }

  uint32_t num_points = 0;
  ifs.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
  if (!ifs) {
    std::cerr << "[ERROR] Failed to read number of points" << std::endl;
    return false;
  }

  cloud.clear();
  cloud.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    float data[3];
    ifs.read(reinterpret_cast<char*>(data), sizeof(float) * 3);
    if (!ifs) {
      std::cerr << "[ERROR] Failed to read point data at index " << i << std::endl;
      return false;
    }
    cloud[i] = Eigen::Vector3f(data[0], data[1], data[2]);
  }

  ifs.close();

  return true;
}

bool ReadPointCloud(
  const std::string& fname,
  PointCloud3f& cloud,
  std::vector<float>& intensities) {
  std::ifstream ifs(fname, std::ios::binary);
  if (!ifs) {
    std::cerr << "[ERROR] Cannot open file for reading: " << fname << std::endl;
    return false;
  }

  uint32_t num_points = 0;
  ifs.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
  if (!ifs) {
    std::cerr << "[ERROR] Failed to read number of points" << std::endl;
    return false;
  }

  cloud.clear();
  intensities.clear();
  cloud.resize(num_points);
  intensities.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    float data[4];
    ifs.read(reinterpret_cast<char*>(data), sizeof(float) * 4);
    if (!ifs) {
      std::cerr << "[ERROR] Failed to read point data at index " << i << std::endl;
      return false;
    }
    cloud[i] = Eigen::Vector3f(data[0], data[1], data[2]);
    intensities[i] = data[3];
  }

  ifs.close();

  return true;
}

bool WriteBinaryPCD(
  const std::string& fname,
  const PointCloud3f& cloud) {
  std::ofstream ofs(fname, std::ios::binary);
  if (!ofs) {
    std::cerr << "[ERROR] Cannot open file: " << fname << std::endl;
    return false;
  }

  const size_t num_points = cloud.size();
  const std::string header =
    "VERSION .7\n"
    "FIELDS x y z\n"
    "SIZE 4 4 4\n"
    "TYPE F F F\n"
    "COUNT 1 1 1\n"
    "WIDTH " + std::to_string(num_points) + "\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS " + std::to_string(num_points) + "\n"
    "DATA binary\n";

  ofs.write(header.c_str(), header.size());

  for (size_t i = 0; i < num_points; ++i) {
    const float data[3] = {cloud[i].x(), cloud[i].y(), cloud[i].z()};
    ofs.write(reinterpret_cast<const char*>(data), sizeof(float) * 3);
  }

  ofs.close();

  return true;
}

bool WriteBinaryPCD(
  const std::string& fname,
  const PointCloud3f& cloud,
  const std::vector<float>& intensities) {
  if (cloud.size() != intensities.size()) {
    std::cerr << "[ERROR] cloud size != intensity size" << std::endl;
    return false;
  }

  std::ofstream ofs(fname, std::ios::binary);
  if (!ofs) {
    std::cerr << "[ERROR] Cannot open file: " << fname << std::endl;
    return false;
  }

  const size_t num_points = cloud.size();
  const std::string header =
    "VERSION .7\n"
    "FIELDS x y z intensity\n"
    "SIZE 4 4 4 4\n"
    "TYPE F F F F\n"
    "COUNT 1 1 1 1\n"
    "WIDTH " + std::to_string(num_points) + "\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS " + std::to_string(num_points) + "\n"
    "DATA binary\n";

  ofs.write(header.c_str(), header.size());

  for (size_t i = 0; i < num_points; ++i) {
    const float data[4] = {cloud[i].x(), cloud[i].y(), cloud[i].z(), intensities[i]};
    ofs.write(reinterpret_cast<const char*>(data), sizeof(float) * 4);
  }

  ofs.close();

  return true;
}

bool ReadPCD(
  const std::string& fname,
  PointCloud3f& cloud,
  std::vector<float>& intensities) {
  std::ifstream ifs(fname, std::ios::binary);
  if (!ifs) {
    std::cerr << "[ERROR] Cannot open file: " << fname << std::endl;
    return false;
  }

  std::string line;
  size_t num_points = 0;
  bool has_intensity = false;
  std::string data_type;
  size_t header_end_pos = 0;

  while (std::getline(ifs, line)) {
    if (line.substr(0, 6) == "FIELDS") {
      std::istringstream ss(line.substr(7));
      std::string field;
      while (ss >> field) {
        if (field == "intensity") {
          has_intensity = true;
        }
      }
    } else if (line.substr(0, 6) == "POINTS") {
      num_points = std::stoul(line.substr(7));
    } else if (line.substr(0, 4) == "DATA") {
      std::istringstream ss(line);
      std::string token;
      ss >> token >> data_type;
      header_end_pos = static_cast<size_t>(ifs.tellg());
      break;
    }
  }

  if (num_points == 0 ||
      (data_type != "binary" && data_type != "ascii" && data_type != "binary_compressed")) {
    std::cerr << "[ERROR] Unsupported or malformed PCD file." << std::endl;
    return false;
  }

  cloud.clear();
  intensities.clear();
  cloud.reserve(num_points);
  if (has_intensity) {
    intensities.reserve(num_points);
  }

  if (data_type == "binary") {
    ifs.seekg(header_end_pos);
    for (size_t i = 0; i < num_points; ++i) {
      float xyz[3];
      ifs.read(reinterpret_cast<char*>(xyz), sizeof(float) * 3);
      cloud.emplace_back(xyz[0], xyz[1], xyz[2]);

      if (has_intensity) {
        float intensity;
        ifs.read(reinterpret_cast<char*>(&intensity), sizeof(float));
        intensities.push_back(intensity);
      }
    }

  } else if (data_type == "binary_compressed") {
    if (!ParseBinaryCompressedPCD(fname, cloud, intensities)) {
      return false;
    }
  } else if (data_type == "ascii") {
    std::string data_line;
    while (std::getline(ifs, data_line)) {
      std::istringstream ss(data_line);
      float x, y, z;
      ss >> x >> y >> z;
      cloud.emplace_back(x, y, z);

      if (has_intensity) {
        float intensity;
        ss >> intensity;
        intensities.push_back(intensity);
      }
    }
  }

  return true;
}

} // namespace pslam
