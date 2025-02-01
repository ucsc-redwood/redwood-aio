#include <volk.h>

#include <iostream>
#include <vector>

int main() {
  // Initialize Volk
  if (volkInitialize() != VK_SUCCESS) {
    std::cerr << "Failed to initialize Volk!" << std::endl;
    return -1;
  }

  // Initialize Vulkan
  VkInstance instance;
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Warp Size Query";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    std::cerr << "Failed to create Vulkan instance!" << std::endl;
    return -1;
  }

  // Load instance-level functions
  volkLoadInstance(instance);

  // Enumerate physical devices
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    std::cerr << "No Vulkan-compatible devices found!" << std::endl;
    return -1;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  // Query warp size for each device
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties2 deviceProperties2 = {};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    VkPhysicalDeviceSubgroupProperties subgroupProperties = {};
    subgroupProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    deviceProperties2.pNext = &subgroupProperties;

    vkGetPhysicalDeviceProperties2(device, &deviceProperties2);

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    std::cout << "Device: " << deviceProperties.deviceName << std::endl;
    std::cout << "  Subgroup Size (Warp Size): "
              << subgroupProperties.subgroupSize << std::endl;
  }

  // Cleanup
  vkDestroyInstance(instance, nullptr);

  return 0;
}
