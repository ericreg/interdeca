set(MODULE_NAME InterDeCANext)
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_NAME}.py
  RESOURCES Resources/Icons/InterDeCANext.png
)

# Keep real packages in the module import directory instead of concatenating source files.
install(DIRECTORY interdeca_slicer
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  COMPONENT Runtime
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/interdeca
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  COMPONENT Runtime
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE)

# The build tree is also loadable before packaging, using the same package structure.
file(COPY interdeca_slicer DESTINATION ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/src/interdeca DESTINATION ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR})
