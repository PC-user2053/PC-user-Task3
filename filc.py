import pandas as pd
import os

def get_test_requirements():
    """Returns a list of unique requirements extracted from the original 150 pairs."""
    requirement_pairs = [
        ("The vehicle must achieve a fuel efficiency of at least 50 km/l.", "Users should be able to turn off the headlight manually.", "Sustainability Conflict"),
        ("The bike must have a top speed of 120 km/h.", "The vehicle should be cost-effective and use affordable materials.", "Regulatory Conflict"),
        ("The vehicle should include tubeless tires for durability and safety.", "The two-wheeler should have a maximum curb weight of 120 kg.", "Material Conflict"),
        ("Users should be able to turn off the headlight manually.", "The frame should be made of lightweight aluminum for efficiency.", "Technology Conflict"),
        ("The vehicle should be affordable with a base price under $1,500.", "The vehicle should be cost-effective and use affordable materials.", "No Conflict"),
        ("The engine should have a fuel injection system for optimized combustion.", "The vehicle design should be compact and lightweight.", "Material Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The engine must comply with strict noise restrictions.", "Resource Conflict"),
        ("The vehicle must meet Euro 6 emission standards.", "The bike should support over-the-air (OTA) software updates.", "Sustainability Conflict"),
        ("The bike must have a top speed of 120 km/h.", "The frame should be made of lightweight aluminum for efficiency.", "Cost Conflict"),
        ("The vehicle must achieve a fuel efficiency of at least 50 km/l.", "The bike should include an always-on headlight for safety compliance.", "Sustainability Conflict"),
        ("The vehicle should include tubeless tires for durability and safety.", "The vehicle should incorporate modular components for easy customization.", "Performance Conflict"),
        ("The vehicle should be cost-effective and use affordable materials.", "The two-wheeler should have a maximum curb weight of 120 kg.", "No Conflict"),
        ("The vehicle design should be compact and lightweight.", "The wheels should be spoked to maintain classic aesthetics.", "Material Conflict"),
        ("The vehicle should include tubeless tires for durability and safety.", "The bike should include an always-on headlight for safety compliance.", "Performance Conflict"),
        ("The engine must comply with strict noise restrictions.", "The engine should have a minimum power output of 25 HP.", "Design Conflict"),
        ("The two-wheeler should have a maximum curb weight of 120 kg.", "The engine should have a fuel injection system for optimized combustion.", "Compliance Conflict"),
        ("The bike should have a storage compartment for a full-size helmet.", "The vehicle should support fast charging to reach 80% in 30 minutes.", "Performance Conflict"),
        ("The instrument cluster should use analog dials to reduce complexity.", "The wheels should be spoked to maintain classic aesthetics.", "Regulatory Conflict"),
        ("The two-wheeler should use biodegradable materials where possible.", "The engine should have a minimum power output of 25 HP.", "Cost Conflict"),
        ("The seat should be made of premium leather for comfort.", "The onboard computer should be minimal to reduce electronic dependencies.", "Material Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The display panel must be fully digital with GPS navigation.", "Design Conflict"),
        ("The vehicle must include a high-capacity battery for extended electric range.", "The bike should be resistant to extreme weather conditions.", "Performance Conflict"),
        ("The engine should have a fuel injection system for optimized combustion.", "The bike should be resistant to extreme weather conditions.", "Material Conflict"),
        ("The vehicle should have a minimalistic dashboard design.", "The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Sustainability Conflict"),
        ("The seat should be made of premium leather for comfort.", "The instrument cluster should use analog dials to reduce complexity.", "No Conflict"),
        ("The vehicle should be fully electric.", "The bike must have a top speed of 120 km/h.", "No Conflict"),
        ("The vehicle should include tubeless tires for durability and safety.", "The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Cost Conflict"),
        ("The bike should include an always-on headlight for safety compliance.", "The instrument cluster should use analog dials to reduce complexity.", "Material Conflict"),
        ("The vehicle should have a minimalistic dashboard design.", "The frame should be made of lightweight aluminum for efficiency.", "Material Conflict"),
        ("The engine must comply with strict noise restrictions.", "The vehicle must achieve a fuel efficiency of at least 50 km/l.", "Performance Conflict"),
        ("The vehicle should be fully electric.", "The onboard computer should be minimal to reduce electronic dependencies.", "Technology Conflict"),
        ("The bike should be resistant to extreme weather conditions.", "The vehicle must meet Euro 6 emission standards.", "Material Conflict"),
        ("The vehicle should have a minimalistic dashboard design.", "The wheels should be spoked to maintain classic aesthetics.", "Technology Conflict"),
        ("The bike should be resistant to extreme weather conditions.", "The bike must have a top speed of 120 km/h.", "Performance Conflict"),
        ("The engine must comply with strict noise restrictions.", "The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Performance Conflict"),
        ("The onboard computer should be minimal to reduce electronic dependencies.", "The bike should have a storage compartment for a full-size helmet.", "Performance Conflict"),
        ("The seat should be made of premium leather for comfort.", "The vehicle should be cost-effective and use affordable materials.", "Technology Conflict"),
        ("The engine should have a minimum power output of 25 HP.", "The engine should have a fuel injection system for optimized combustion.", "Compliance Conflict"),
        ("The two-wheeler should use biodegradable materials where possible.", "The engine should have a fuel injection system for optimized combustion.", "Cost Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The bike should be resistant to extreme weather conditions.", "Sustainability Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The wheels should be spoked to maintain classic aesthetics.", "Material Conflict"),
        ("The display panel must be fully digital with GPS navigation.", "The vehicle should have a minimalistic dashboard design.", "Sustainability Conflict"),
        ("The engine should have a fuel injection system for optimized combustion.", "The vehicle should be affordable with a base price under $1,500.", "Design Conflict"),
        ("The engine must comply with strict noise restrictions.", "The two-wheeler must have a rearview camera for safety.", "Cost Conflict"),
        ("Users should be able to turn off the headlight manually.", "Production costs should remain below market average for affordability.", "Compliance Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The wheels should be spoked to maintain classic aesthetics.", "Regulatory Conflict"),
        ("The vehicle should be cost-effective and use affordable materials.", "The display panel must be fully digital with GPS navigation.", "Contradiction"),
        ("Production costs should remain below market average for affordability.", "The instrument cluster should use analog dials to reduce complexity.", "Technology Conflict"),
        ("The frame should be made of lightweight aluminum for efficiency.", "The braking system should use ABS for enhanced safety.", "No Conflict"),
        ("The two-wheeler must have a rearview camera for safety.", "Durability must be ensured for at least 10 years of use.", "Compliance Conflict"),
        ("The wheels should be spoked to maintain classic aesthetics.", "The frame should be made of lightweight aluminum for efficiency.", "Resource Conflict"),
        ("The engine design should prioritize performance over emission constraints.", "Users should be able to turn off the headlight manually.", "Design Conflict"),
        ("The system should minimize power consumption for extended battery life.", "The engine design should prioritize performance over emission constraints.", "Regulatory Conflict"),
        ("The vehicle should incorporate modular components for easy customization.", "The two-wheeler must have a rearview camera for safety.", "Contradiction"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The bike should support over-the-air (OTA) software updates.", "Regulatory Conflict"),
        ("The braking system should use ABS for enhanced safety.", "The vehicle should incorporate modular components for easy customization.", "Contradiction"),
        ("Users should be able to turn off the headlight manually.", "The vehicle should incorporate modular components for easy customization.", "Sustainability Conflict"),
        ("The two-wheeler should have a maximum curb weight of 120 kg.", "The frame should be made of lightweight aluminum for efficiency.", "Technology Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The vehicle should be fully electric.", "No Conflict"),
        ("The instrument cluster should use analog dials to reduce complexity.", "The battery should use standard charging methods to maintain longevity.", "Material Conflict"),
        ("The engine design should prioritize performance over emission constraints.", "The frame should be made of lightweight aluminum for efficiency.", "No Conflict"),
        ("The vehicle should incorporate modular components for easy customization.", "The two-wheeler should use biodegradable materials where possible.", "Cost Conflict"),
        ("Production costs should remain below market average for affordability.", "The braking system should use ABS for enhanced safety.", "Technology Conflict"),
        ("Users should be able to turn off the headlight manually.", "The engine design should prioritize performance over emission constraints.", "Regulatory Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The engine should have a minimum power output of 25 HP.", "Design Conflict"),
        ("The vehicle should be fully electric.", "The vehicle should be cost-effective and use affordable materials.", "Contradiction"),
        ("The vehicle should have a minimalistic dashboard design.", "Durability must be ensured for at least 10 years of use.", "No Conflict"),
        ("The seat should be made of premium leather for comfort.", "The engine design should prioritize performance over emission constraints.", "Sustainability Conflict"),
        ("Production costs should remain below market average for affordability.", "The vehicle should include tubeless tires for durability and safety.", "Performance Conflict"),
        ("The instrument cluster should use analog dials to reduce complexity.", "The onboard computer should be minimal to reduce electronic dependencies.", "Sustainability Conflict"),
        ("The vehicle design should be compact and lightweight.", "The instrument cluster should use analog dials to reduce complexity.", "Compliance Conflict"),
        ("The onboard computer should be minimal to reduce electronic dependencies.", "The vehicle should support fast charging to reach 80% in 30 minutes.", "Material Conflict"),
        ("The vehicle must meet Euro 6 emission standards.", "The vehicle must achieve a fuel efficiency of at least 50 km/l.", "Regulatory Conflict"),
        ("The two-wheeler must have a rearview camera for safety.", "The vehicle design should be compact and lightweight.", "No Conflict"),
        ("The vehicle should incorporate modular components for easy customization.", "The bike should have a storage compartment for a full-size helmet.", "Design Conflict"),
        ("The two-wheeler should have a maximum curb weight of 120 kg.", "The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Design Conflict"),
        ("The engine design should prioritize performance over emission constraints.", "The bike should include an always-on headlight for safety compliance.", "Cost Conflict"),
        ("The vehicle should be fully electric.", "The battery should use standard charging methods to maintain longevity.", "Performance Conflict"),
        ("The instrument cluster should use analog dials to reduce complexity.", "The vehicle must achieve a fuel efficiency of at least 50 km/l.", "Cost Conflict"),
        ("The seat should be made of premium leather for comfort.", "The vehicle must meet Euro 6 emission standards.", "Technology Conflict"),
        ("The bike should include an always-on headlight for safety compliance.", "The vehicle should incorporate modular components for easy customization.", "Performance Conflict"),
        ("The vehicle design should be compact and lightweight.", "The engine design should prioritize performance over emission constraints.", "Contradiction"),
        ("The frame should be made of lightweight aluminum for efficiency.", "The battery should use standard charging methods to maintain longevity.", "Regulatory Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The wheels should be spoked to maintain classic aesthetics.", "Regulatory Conflict"),
        ("The vehicle design should be compact and lightweight.", "The vehicle must include a high-capacity battery for extended electric range.", "Technology Conflict"),
        ("The display panel must be fully digital with GPS navigation.", "The bike should have a storage compartment for a full-size helmet.", "Cost Conflict"),
        ("The braking system should use ABS for enhanced safety.", "The engine should have a fuel injection system for optimized combustion.", "Performance Conflict"),
        ("The vehicle must include a high-capacity battery for extended electric range.", "The vehicle should support fast charging to reach 80% in 30 minutes.", "Regulatory Conflict"),
        ("The engine should have a fuel injection system for optimized combustion.", "The seat should be made of premium leather for comfort.", "Performance Conflict"),
        ("The onboard computer should be minimal to reduce electronic dependencies.", "The bike should support over-the-air (OTA) software updates.", "Technology Conflict"),
        ("The instrument cluster should use analog dials to reduce complexity.", "The battery should use standard charging methods to maintain longevity.", "Regulatory Conflict"),
        ("The onboard computer should be minimal to reduce electronic dependencies.", "The bike should be resistant to extreme weather conditions.", "Cost Conflict"),
        ("The bike should have a storage compartment for a full-size helmet.", "The bike should include an always-on headlight for safety compliance.", "Compliance Conflict"),
        ("The bike must have a top speed of 120 km/h.", "The vehicle design should be compact and lightweight.", "Compliance Conflict"),
        ("The vehicle must meet Euro 6 emission standards.", "Users should be able to turn off the headlight manually.", "No Conflict"),
        ("The bike should have a storage compartment for a full-size helmet.", "The system should minimize power consumption for extended battery life.", "Resource Conflict"),
        ("The vehicle must include a high-capacity battery for extended electric range.", "The two-wheeler should have a maximum curb weight of 120 kg.", "Performance Conflict"),
        ("The bike should include an always-on headlight for safety compliance.", "The vehicle should incorporate modular components for easy customization.", "Cost Conflict"),
        ("The vehicle should be cost-effective and use affordable materials.", "Production costs should remain below market average for affordability.", "Sustainability Conflict"),
        ("The two-wheeler should use biodegradable materials where possible.", "The vehicle should incorporate modular components for easy customization.", "Technology Conflict"),
        ("The braking system should use ABS for enhanced safety.", "The engine should have a fuel injection system for optimized combustion.", "Technology Conflict"),
        ("The two-wheeler must have a rearview camera for safety.", "The bike should include an always-on headlight for safety compliance.", "Cost Conflict"),
        ("The bike should include an always-on headlight for safety compliance.", "The vehicle must include a high-capacity battery for extended electric range.", "Resource Conflict"),
        ("The seat should be made of premium leather for comfort.", "The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Performance Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The vehicle should be cost-effective and use affordable materials.", "Design Conflict"),
        ("The display panel must be fully digital with GPS navigation.", "The vehicle should support fast charging to reach 80% in 30 minutes.", "Material Conflict"),
        ("The braking system should use ABS for enhanced safety.", "The seat should be made of premium leather for comfort.", "Regulatory Conflict"),
        ("The vehicle should incorporate modular components for easy customization.", "The vehicle should support fast charging to reach 80% in 30 minutes.", "Material Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "The vehicle should include tubeless tires for durability and safety.", "Performance Conflict"),
        ("The vehicle should support smart connectivity via Bluetooth and Wi-Fi.", "Production costs should remain below market average for affordability.", "Performance Conflict"),
        ("The engine design should prioritize performance over emission constraints.", "The vehicle should incorporate modular components for easy customization.", "No Conflict"),
        ("The vehicle must meet Euro 6 emission standards.", "The bike should support over-the-air (OTA) software updates.", "Contradiction"),
        ("The bike should include an always-on headlight for safety compliance.", "The vehicle design should be compact and lightweight.", "Technology Conflict"),
        ("The display panel must be fully digital with GPS navigation.", "The bike must have a top speed of 120 km/h.", "Material Conflict"),
        ("The onboard computer should be minimal to reduce electronic dependencies.", "Production costs should remain below market average for affordability.", "Resource Conflict"),
        ("The vehicle must achieve a fuel efficiency of at least 50 km/l.", "The vehicle must meet Euro 6 emission standards.", "Material Conflict"),
        ("The vehicle must meet Euro 6 emission standards.", "The display panel must be fully digital with GPS navigation.", "Contradiction"),
        ("The bike must have a top speed of 120 km/h.", "Production costs should remain below market average for affordability.", "Cost Conflict"),
        ("The two-wheeler must have a rearview camera for safety.", "The instrument cluster should use analog dials to reduce complexity.", "Regulatory Conflict"),
        ("The battery should use standard charging methods to maintain longevity.", "The vehicle should be cost-effective and use affordable materials.", "Cost Conflict"),
        ("The vehicle must include a high-capacity battery for extended electric range.", "The vehicle must meet Euro 6 emission standards.", "Regulatory Conflict"),
        ("The vehicle should have a minimalistic dashboard design.", "The vehicle must include a high-capacity battery for extended electric range.", "Resource Conflict"),
        ("The wheels should be spoked to maintain classic aesthetics.", "The braking system should use ABS for enhanced safety.", "Performance Conflict"),
        ("The bike must have a top speed of 120 km/h.", "The seat should be made of premium leather for comfort.", "Technology Conflict"),
        ("The braking system should use ABS for enhanced safety.", "The instrument cluster should use analog dials to reduce complexity.", "Material Conflict"),
        ("The vehicle should be fully electric.", "Production costs should remain below market average for affordability.", "Performance Conflict"),
        ("The bike should support over-the-air (OTA) software updates.", "The wheels should be spoked to maintain classic aesthetics.", "Technology Conflict")
    ]
    
    # Extract all unique requirements from Requirement_1 and Requirement_2
    all_requirements = set()
    for req1, req2, _ in requirement_pairs:
        all_requirements.add(req1)
        all_requirements.add(req2)
    
    return list(all_requirements)

if __name__ == "__main__":
    # Get the unique requirements
    data = get_test_requirements()
    
    # Create a DataFrame with a single "Requirements" column
    df = pd.DataFrame(data, columns=["Requirements"])
    
    # Ensure Test_data directory exists
    os.makedirs("Test_data", exist_ok=True)
    
    # Save to CSV in Test_data/data.csv
    csv_path = os.path.join("Test_data", "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")