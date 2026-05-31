import yaml
with open('config/equipment_classes.yaml') as f:
    data = yaml.safe_load(f)
print(f"Required by default: {data.get('required_by_default', [])}")
print(f"Names: {data.get('names', {})}")
