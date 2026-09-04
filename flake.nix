{
  description = "SIH26168 - AI-ML Intelligent Dead Reckoning dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            android_sdk.accept_license = true;
          };
        };

        python = pkgs.python312.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          pytest
          pip
          virtualenv
        ]);

        androidSdk = pkgs.androidenv.composeAndroidPackages {
          platformVersions = [ "34" ];
          buildToolsVersions = [ "34.0.0" ];
          platformToolsVersion = "37.0.1";
          includeEmulator = false;
          includeSources = false;
          includeSystemImages = false;
        };

        nativeLibraries = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          libffi
          openssl
          proj
          geos
          gdal
          libspatialite
          sqlite
        ];
      in {
        devShells = {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.pkg-config
              pkgs.gcc
              pkgs.cmake
              pkgs.git-lfs
              pkgs.which
              pkgs.jq
            ];

            buildInputs = nativeLibraries;

            env = {
              MPLBACKEND = "Agg";
              PIP_DISABLE_PIP_VERSION_CHECK = "1";
              PYTHONNOUSERSITE = "1";
              PROJ_NETWORK = "OFF";
              GDAL_DATA = "${pkgs.gdal}/share/gdal";
              PROJ_LIB = "${pkgs.proj}/share/proj";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeLibraries;
            };

            shellHook = ''
              set -e
              venv="$PWD/.venv"

              if [ ! -d "$venv" ]; then
                echo ">> creating .venv with system-site-packages"
                python -m venv --system-site-packages "$venv"
              fi

              export PATH="$venv/bin:$PATH"
              export VIRTUAL_ENV="$venv"
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"

              if ! python -c "import ahrs, leuvenmapmatching, osmnx, torch" >/dev/null 2>&1; then
                echo ">> installing Python packages not supplied by nixpkgs"
                python -m pip install --disable-pip-version-check --no-input \
                  ahrs==0.3.1 leuvenmapmatching==1.1.4 osmnx==1.9.4
                python -m pip install --disable-pip-version-check --no-input \
                  torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
              fi

              echo ">> SIH26168 environment ready"
              echo "   Python: $(python --version)"
              echo "   Run:    pytest python/tests/ -q"
            '';
          };

          android = pkgs.mkShell {
            packages = with pkgs; [
              android-studio
              android-tools
              jdk17
              gradle
              kotlin
              unzip
              zip
            ];

            buildInputs = [ androidSdk.androidsdk ];

            env = {
              ANDROID_HOME = "${androidSdk.androidsdk}/libexec/android-sdk";
              ANDROID_SDK_ROOT = "${androidSdk.androidsdk}/libexec/android-sdk";
              JAVA_HOME = "${pkgs.jdk17}";
            };

            shellHook = ''
              echo ">> Android shell ready"
              echo "   Android SDK: $ANDROID_HOME"
              echo "   Platform: android-34"
              echo "   Build tools: 34.0.0"
            '';
          };
        };
      });
}
