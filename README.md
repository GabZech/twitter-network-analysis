# twitter-network-analysis

## Installation

Since this project uses graph-tool and this package is only available for Linux (as of April 2023), we provide installation instructions for Linux only.

### Conda environment
Install from the `environment.yml` file using conda:
`conda env create -f environment.yml`

### Install Quarto

On Linux:
1. `export QUARTO_VERSION="1.3.309"`
2. `sudo curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb`
3. `sudo apt-get install gdebi-core` - not needed if you already have gdebi
4. `sudo gdebi quarto-linux-amd64.deb`
5. `/usr/local/bin/quarto check` - check is your Quarto installation was successful


If you need any help, see (Quarto's installation instructions)[https://docs.posit.co/resources/install-quarto/#quarto-deb-file-install]

## Render Quarto website

On terminal in your project folder: `quarto render`

On VSCode: Press `Ctrl + Shift + P` and type `Quarto: Render Project`

Then simply push the changes to the `main` branch and the website will be automatically deployed.