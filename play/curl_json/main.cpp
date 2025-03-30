#include <curl/curl.h>

#include <iostream>
#include <string>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
  size_t totalSize = size * nmemb;
  output->append((char*)contents, totalSize);
  return totalSize;
}

int main() {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to init curl\n";
    return 1;
  }

  std::string response;
  curl_easy_setopt(curl, CURLOPT_URL, "http://192.168.1.95/hey.json");

  // Set a timeout (seconds) - increasing the connect timeout
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);         // Max time for request
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);  // Increased connect timeout

  // Enable verbose mode for debugging
  // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

  // Optional: follow redirects
  // curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  // Add DNS resolver timeout
  // curl_easy_setopt(curl, CURLOPT_DNS_CACHE_TIMEOUT, 120L);

  // Print information about the request
  std::cout << "Connecting to: http://192.168.1.95/hey.json\n";

  // Optional: Try to use IPv4 only to avoid IPv6 issues
  curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

  // Rest of your code
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  std::cout << "Performing request...\n";

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "curl error: " << curl_easy_strerror(res) << "\n";

    // Print more detailed error information
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code > 0) {
      std::cerr << "HTTP response code: " << http_code << "\n";
    }
  } else {
    std::cout << "Response:\n" << response << "\n";
  }

  curl_easy_cleanup(curl);
  return 0;
}
