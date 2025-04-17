cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BrainBiopsy)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BrainBiopsy")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Muhammed Fatih Talu (Inonu University), Mustafa Şahin (Hasan Kalyoncu University)")
set(EXTENSION_DESCRIPTION "BrainBiopsy extension has capabilities to do:
 brain extraction from Mri volumes 
 tumor extraction from t1 Mri volumes
 vessel extraction from mra volumes
 risk calculation for biopsy operation
 ")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/BrainBiopsy.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/BrainBiopsy/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BrainBiopsy)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
