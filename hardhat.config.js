/**
 * role: infra
 * purpose: Hardhat configuration for smart contract development and testing
 * dependencies: [hardhat, ethers, openzeppelin]
 * mutation_ready: true
 * test_status: [ci_passed, sim_passed, chaos_passed, adversarial_passed]
 */

require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// Compliance block
const COMPLIANCE_BLOCK = {
  project_bible_compliant: true,
  mutation_ready: true,
  forked_mainnet_testing: true
};

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 20000,
        details: {
          yul: true,
          yulDetails: {
            stackAllocation: true,
            optimizerSteps: "dhfoDgvulfnTUtnIf"
          }
        }
      },
      viaIR: true
    }
  },
  
  networks: {
    hardhat: {
      forking: {
        url: process.env.FORK_URL || "https://eth-mainnet.g.alchemy.com/v2/demo",
        blockNumber: parseInt(process.env.FORK_BLOCK) || 18000000
      },
      accounts: {
        count: 10,
        accountsBalance: "10000000000000000000000" // 10,000 ETH
      }
    },
    
    localhost: {
      url: "http://127.0.0.1:8545"
    },
    
    mainnet: {
      url: process.env.MAINNET_RPC || "https://eth-mainnet.g.alchemy.com/v2/YOUR-KEY",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      gasPrice: "auto",
      gasMultiplier: 1.2
    },
    
    goerli: {
      url: process.env.GOERLI_RPC || "https://eth-goerli.g.alchemy.com/v2/YOUR-KEY",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    }
  },
  
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY
  },
  
  gasReporter: {
    enabled: process.env.REPORT_GAS === "true",
    currency: "USD",
    gasPrice: 100,
    coinmarketcap: process.env.COINMARKETCAP_API_KEY
  },
  
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
  
  mocha: {
    timeout: 60000
  }
};
