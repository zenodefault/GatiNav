{
  description = "SIH26168 — AI-ML Intelligent Dead Reckoning dev environment";

  inputs = {
    # nixos-unstable for recent torch; pin to your system's channel if preferred
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.android_sdk.accept_license = true;
        };

        androidSdk = pkgs.androidenv.composeAndroidPackages {
          platformVersions = [ "34" ];
          buildToolsVersions = [ "34.0.0" ];
          platformToolsVersion = "37.0.1";
          includeEmulator = false;
          includeSources = false;
          includeSystemImages = false;
        };

        # Keep binary/scientific dependencies in nixpkgs so Nix supplies
        # patched CPU builds and their native runtime libraries.
        # nixos-unstable currently exposes Python 3.12; the application
        # remains compatible with the Python 3.10 requirements contract.
        pythonWithSci = pkgs.python312.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          pytest
          torch
          osmnx
          jupyter
          notebook
          ipykernel
          pip
          virtualenv
        ]);

        nativeRuntime = with pkgs; [
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
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = [
              pythonWithSci
              pkgs.uv
              pkgs.pkg-config
              pkgs.gcc
              pkgs.cmake
              pkgs.which
              pkgs.jq
            ];

            buildInputs = nativeRuntime;

            env = {
              MPLBACKEND = "Agg";
              PIP_DISABLE_PIP_VERSION_CHECK = "1";
              PYTHONNOUSERSITE = "1";
              PROJ_NETWORK = "OFF";
            };

            shellHook = ''
              if [ ! -d .venv ]; then
                echo ">> creating .venv (one-time) with system-site-packages"
                python -m venv --system-site-packages .venv
              fi
              source .venv/bin/activate

              # These packages are pure Python gaps when unavailable in the
              # selected nixpkgs revision. Keep installation non-interactive.
              if ! python -c "import leuvenmapmatching" 2>/dev/null \
                 || ! python -c "import ahrs" 2>/dev/null; then
                echo ">> installing gap packages (ahrs, leuvenmapmatching)"
                python -m pip install --disable-pip-version-check --no-input \
                  ahrs==0.3.1 leuvenmapmatching==1.1.27
              fi

              echo ">> SIH26168 env ready"
              echo "   python : $(python --version)"
              echo "   numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
              echo "   torch  : $(python -c 'import torch; print(torch.__version__)')"
              echo "   osmnx  : $(python -c 'import osmnx; print(osmnx.__version__)')"
              echo "   run    : pytest python/tests/ -q"
            '';
          };

          android = pkgs.mkShell {
            packages = with pkgs; [
              android-studio
              android-tools
              jdk17
              gradle
              kotlin
              git
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
              echo "   Android SDK : $ANDROID_HOME"
              echo "   platform    : android-34"
              echo "   build tools : 34.0.0"
              echo "   ensure udev rules + adbusers group for phone access"
            '';
          };
        };
      });
}
