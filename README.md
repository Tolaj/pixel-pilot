# list what's available
## python -m perception --list

# quick parse with one provider
## python -m perception --provider moondream

# benchmark all three on the same screenshot, 3 runs each
## python -m perception --benchmark --runs 3

# benchmark specific two
## python -m perception --benchmark ax moondream

# use a saved screenshot instead of live capture
## python -m perception --provider omniparser --image test_screenshot.png
