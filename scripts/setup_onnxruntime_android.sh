#!/bin/bash
# Download ONNX Runtime native libraries for Android
# Run this script after cloning the repo

set -e

ONNX_VERSION="1.16.3"
AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNX_VERSION}/onnxruntime-android-${ONNX_VERSION}.aar"
JNILIBS_DIR="apps/mobile_flutter/android/app/src/main/jniLibs"

echo "Downloading ONNX Runtime ${ONNX_VERSION} for Android..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download AAR
curl -L -o onnxruntime.aar "$AAR_URL"

# Extract
unzip -q onnxruntime.aar -d extracted

# Copy native libraries
cd -
mkdir -p "${JNILIBS_DIR}/arm64-v8a"
mkdir -p "${JNILIBS_DIR}/armeabi-v7a"
mkdir -p "${JNILIBS_DIR}/x86_64"

cp "${TEMP_DIR}/extracted/jni/arm64-v8a/libonnxruntime.so" "${JNILIBS_DIR}/arm64-v8a/"
cp "${TEMP_DIR}/extracted/jni/armeabi-v7a/libonnxruntime.so" "${JNILIBS_DIR}/armeabi-v7a/"
cp "${TEMP_DIR}/extracted/jni/x86_64/libonnxruntime.so" "${JNILIBS_DIR}/x86_64/"

# Cleanup
rm -rf "$TEMP_DIR"

echo "ONNX Runtime native libraries installed to ${JNILIBS_DIR}"
echo "  arm64-v8a: $(du -h ${JNILIBS_DIR}/arm64-v8a/libonnxruntime.so | cut -f1)"
echo "  armeabi-v7a: $(du -h ${JNILIBS_DIR}/armeabi-v7a/libonnxruntime.so | cut -f1)"
echo "  x86_64: $(du -h ${JNILIBS_DIR}/x86_64/libonnxruntime.so | cut -f1)"
