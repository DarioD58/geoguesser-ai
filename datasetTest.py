from LocationDataset import LocationDataset
import matplotlib.pyplot as plt


dataset = LocationDataset("data.csv", "data")

images, latitude, longitude = dataset[0]

print(f"Latitude is {latitude}")
print(f"Longitude is {longitude}")

plt.imshow(images[0])
plt.show()

datasetNorth = LocationDataset("data.csv", "data", orientation="0")

image, latitude, longitude = datasetNorth[0]

print(f"Latitude is {latitude}")
print(f"Longitude is {longitude}")

plt.imshow(image)
plt.show()