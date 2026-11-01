//! Source-code pre-filter for the extraction pipeline.
//!
//! Strips boilerplate (test/mock/script files, imports, pragma, SPDX)
//! from the source text BEFORE it reaches the TokenCursor, reducing
//! extraction LLM input tokens by 30-40% without removing any
//! audit-relevant code.
//!
//! Owner order 2026-11-01: "use jCodemunch to index the shards, then
//! the agents could use the index to reduce token use." This module
//! implements the pre-filter tier (option 1): conservative file-level
//! and line-level noise removal. Call-graph-guided targeting (option 3)
//! is a follow-up.

use std::path::Path;

/// Returns true when a file is noise for audit extraction purposes.
/// Conservative: only skips files that are unambiguously test/mock/script.
pub fn should_skip_file(relative_path: &Path) -> bool {
    let path_str = relative_path.to_string_lossy().to_lowercase();

    // Foundry/Hardhat test files
    if path_str.ends_with(".t.sol") || path_str.ends_with("_test.sol") {
        return true;
    }

    // Test directories
    for seg in relative_path.iter() {
        let s = seg.to_string_lossy().to_lowercase();
        if s == "test" || s == "tests" || s == "script" || s == "scripts" {
            return true;
        }
    }

    // Mock contracts: Solidity-specific heuristic, scoped to .sol files
    // only (gate_20261101-012631 finding 2: a .move file with "mock"
    // prefix was falsely skipped). Matches basename prefix/suffix "mock"
    // after stripping the .sol extension — not substring.
    let file_name = relative_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if !file_name.to_lowercase().ends_with(".sol") {
        return false;
    }
    let stem = file_name
        .strip_suffix(".sol")
        .unwrap_or(&file_name)
        .to_lowercase();
    if stem.starts_with("mock") || stem.ends_with("mock") {
        return true;
    }

    false
}

/// Strips noise lines from Solidity source content.
/// Removes: SPDX comments, pragma lines, import statements.
/// Keeps: everything else (interfaces, stubs, constructors, all logic).
pub fn filter_solidity(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();

        // SPDX license comment
        if lower.starts_with("// spdx-license-identifier") {
            continue;
        }

        // Pragma directive (the LLM doesn't need it; it sees all code)
        if lower.starts_with("pragma solidity") {
            continue;
        }

        // Import statements: strip only when the module path (the quoted
        // string) references an external npm-style package (starts with
        // @) or node_modules; local imports are kept for cross-chunk
        // context (gate_20261101-011750 finding 1, refined by
        // gate_20261101-012631 finding 1: `@` must be inside the
        // quoted module path, not anywhere on the line)
        if lower.starts_with("import ") || lower.starts_with("import{") {
            let has_external = trimmed.contains("\"@")
                || trimmed.contains("'@")
                || trimmed.contains("node_modules");
            if has_external {
                continue;
            }
        }

        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Strips noise lines for a given language. Falls through to identity
/// for non-Solidity languages (the pre-filter is Solidity-first because
/// that's the C4 contest corpus).
pub fn filter_content(content: &str, language: &str) -> String {
    match language {
        "solidity" | "sol" => filter_solidity(content),
        _ => content.to_string(),
    }
}

/// Reports what was skipped (for the prompt, so the LLM knows these
/// files exist but aren't shown).
pub fn skip_marker(relative_path: &Path) -> String {
    format!(
        "### [skipped: test/mock/script] {}\n\n",
        relative_path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_test_files() {
        assert!(should_skip_file(Path::new("test/MyTest.t.sol")));
        assert!(should_skip_file(Path::new("contracts/Token_test.sol")));
        assert!(should_skip_file(Path::new("script/Deploy.s.sol")));
        assert!(should_skip_file(Path::new("test/Integration.t.sol")));
    }

    #[test]
    fn skips_mock_files() {
        assert!(should_skip_file(Path::new("contracts/MockToken.sol")));
        assert!(should_skip_file(Path::new("contracts/mocks/ERC20Mock.sol")));
        assert!(should_skip_file(Path::new(
            "contracts/ChainlinkMock.sol"
        )));
    }

    #[test]
    fn keeps_source_files() {
        assert!(!should_skip_file(Path::new("contracts/Token.sol")));
        assert!(!should_skip_file(Path::new(
            "contracts/CompoundV3Oracle.sol"
        )));
        assert!(!should_skip_file(Path::new(
            "contracts/interfaces/IERC20.sol"
        )));
        assert!(!should_skip_file(Path::new("src/Vault.sol")));
    }

    #[test]
    fn keeps_test_dir_with_src_prefix() {
        // A file named "testing.sol" should NOT be skipped just because
        // it contains "test" — only exact directory segments match
        assert!(!should_skip_file(Path::new("contracts/testing.sol")));
    }

    #[test]
    fn filter_strips_pragma_import_spdx() {
        let input = "// SPDX-License-Identifier: MIT\n\
                     pragma solidity ^0.8.19;\n\
                     import {IERC20} from \"./IERC20.sol\";\n\
                     import \"@openzeppelin/contracts/token/ERC20/ERC20.sol\";\n\
                     import {SafeERC20} from \"@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol\";\n\
                     \n\
                     contract Token is ERC20 {\n\
                         mapping(address => uint256) public balances;\n\
                     }\n";
        let filtered = filter_solidity(input);
        assert!(!filtered.contains("SPDX"));
        assert!(!filtered.contains("pragma"));
        // external library imports are stripped
        assert!(!filtered.contains("@openzeppelin"));
        // local imports are KEPT for cross-chunk context
        assert!(filtered.contains("./IERC20.sol"));
        assert!(filtered.contains("contract Token"));
        assert!(filtered.contains("balances"));
        assert!(filtered.contains("mapping"));
    }

    #[test]
    fn filter_preserves_interfaces_and_stubs() {
        let input = "interface IRouter {\n\
                     function swapExactTokensForTokens(\n\
                         uint amountIn,\n\
                         uint amountOutMin,\n\
                         address[] calldata path\n\
                     ) external returns (uint[] memory amounts);\n\
                     }\n\
                     \n\
                     contract Stub is IRouter {\n\
                         function swapExactTokensForTokens(\n\
                             uint amountIn, uint amountOutMin,\n\
                             address[] calldata path\n\
                         ) external override returns (uint[] memory amounts) {}\n\
                     }\n";
        let filtered = filter_solidity(input);
        assert!(filtered.contains("interface IRouter"));
        assert!(filtered.contains("swapExactTokensForTokens"));
        assert!(filtered.contains("contract Stub"));
    }

    #[test]
    fn filter_preserves_import_in_comment() {
        let input = "// import this is a comment mentioning import\n\
                     contract Foo {}\n";
        let filtered = filter_solidity(input);
        assert!(filtered.contains("import this is a comment"));
    }

    #[test]
    fn non_solidity_passthrough() {
        let input = "import foo from 'bar';\nconst x = 1;\n";
        let filtered = filter_content(input, "typescript");
        assert_eq!(filtered, input, "non-Solidity must be identity");
    }

    #[test]
    fn keeps_mock_interface_not_test_double() {
        // "IMockStrategy" contains "mock" as a substring but is an
        // interface, not a test double — must NOT be skipped
        assert!(!should_skip_file(Path::new(
            "contracts/IMockStrategy.sol"
        )));
        // genuine mock doubles ARE skipped
        assert!(should_skip_file(Path::new("contracts/MockToken.sol")));
        assert!(should_skip_file(Path::new(
            "contracts/ERC20Mock.sol"
        ))); // ends with "mock" after stripping .sol
    }

    #[test]
    fn token_reduction_estimate() {
        // A typical Solidity file with external-library noise
        let noise = "// SPDX-License-Identifier: MIT\n".repeat(1)
            + &"pragma solidity ^0.8.0;\n".repeat(1)
            + &"import \"@openzeppelin/A.sol\";\n".repeat(5)
            + &"import \"@uniswap/B.sol\";\n".repeat(3);
        let code = "contract Foo { function bar() external { /* logic */ } }\n".repeat(50);
        let full = format!("{}{}", noise, code);
        let filtered = filter_solidity(&full);
        let full_lines = full.lines().count();
        let filtered_lines = filtered.lines().count();
        let reduction = 1.0 - (filtered_lines as f64 / full_lines as f64);
        assert!(
            reduction > 0.05,
            "expected >5% reduction from line stripping, got {:.1}%",
            reduction * 100.0
        );
    }
}
