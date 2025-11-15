// token_map.cpp

#include "token_map.hpp"

// Constant for padding token ID
const input_ids_type PAD_TOKEN_ID = 0;

std::unordered_map<std::string, input_ids_type> load_token_map() {
	std::unordered_map<std::string, input_ids_type> tokenMap;

	// Total symbols: 178 (Indices 0 through 177, based on ONNX Config "vocab")

	// Punctuation and Symbols
	tokenMap["_"] = PAD_TOKEN_ID;
	tokenMap[";"] = 1;
	tokenMap[":"] = 2;
	tokenMap[","] = 3;
	tokenMap["."] = 4;
	tokenMap["!"] = 5;
	tokenMap["?"] = 6;
	tokenMap["\u2014"] = 9;  // EM DASH (—)
	tokenMap["\u2026"] = 10; // HORIZONTAL ELLIPSIS (…)
	tokenMap["\""] = 11;
	tokenMap["("] = 12;
	tokenMap[")"] = 13;
	tokenMap["\u201C"] = 14; // LEFT DOUBLE QUOTATION MARK (“)
	tokenMap["\u201D"] = 15; // RIGHT DOUBLE QUOTATION MARK (”)
	tokenMap[" "] = 16;  // Space (Word Boundary)

	// IPA and Phonemes
	tokenMap["\u0303"] = 17; // Combining Tilde (Nasalization)
	tokenMap["\u02a3"] = 18; // LATIN SMALL LETTER DZ LIGATURE (ʣ)
	tokenMap["\u02a6"] = 19; // LATIN SMALL LETTER DEZH LIGATURE (ʥ)
	tokenMap["\u02a8"] = 20; // LATIN SMALL LETTER TS LIGATURE (ʦ)
	tokenMap["\u02a7"] = 21; // LATIN SMALL LETTER TESH LIGATURE (ʨ)
	tokenMap["\u1d5d"] = 22; // MODIFIER LETTER SMALL BETA (ᵝ)
	tokenMap["\uab67"] = 23; // Unknown/Rare IPA/Unicode
	tokenMap["A"] = 24;
	tokenMap["I"] = 25;
	tokenMap["O"] = 31;
	tokenMap["Q"] = 33;
	tokenMap["S"] = 35;
	tokenMap["T"] = 36;
	tokenMap["W"] = 39;
	tokenMap["Y"] = 41;
	tokenMap["\u1d4a"] = 42; // MODIFIER LETTER SMALL SCHWA (ᵊ)
	tokenMap["a"] = 43;
	tokenMap["b"] = 44;
	tokenMap["c"] = 45;
	tokenMap["d"] = 46;
	tokenMap["e"] = 47;
	tokenMap["f"] = 48;
	tokenMap["h"] = 50;
	tokenMap["i"] = 51;
	tokenMap["j"] = 52;
	tokenMap["k"] = 53;
	tokenMap["l"] = 54;
	tokenMap["m"] = 55;
	tokenMap["n"] = 56;
	tokenMap["o"] = 57;
	tokenMap["p"] = 58;
	tokenMap["q"] = 59;
	tokenMap["r"] = 60;
	tokenMap["s"] = 61;
	tokenMap["t"] = 62;
	tokenMap["u"] = 63;
	tokenMap["v"] = 64;
	tokenMap["w"] = 65;
	tokenMap["x"] = 66;
	tokenMap["y"] = 67;
	tokenMap["z"] = 68;
	tokenMap["\u0251"] = 69;  // LATIN SMALL LETTER ALPHA (ɑ)
	tokenMap["\u0250"] = 70;  // LATIN SMALL LETTER TURNED A (ɐ)
	tokenMap["\u0252"] = 71;  // LATIN SMALL LETTER TURNED ALPHA (ɒ)
	tokenMap["\u00e6"] = 72;  // LATIN SMALL LETTER AE (æ)
	tokenMap["\u03b2"] = 75;  // GREEK SMALL LETTER BETA (β)
	tokenMap["\u0254"] = 76;  // LATIN SMALL LETTER OPEN O (ɔ)
	tokenMap["\u0255"] = 77;  // LATIN SMALL LETTER C WITH HOOK (ɕ)
	tokenMap["\u00e7"] = 78;  // LATIN SMALL LETTER C WITH CEDILLA (ç)
	tokenMap["\u0256"] = 80;  // LATIN SMALL LETTER D WITH TAIL (ɖ)
	tokenMap["\u00f0"] = 81;  // LATIN SMALL LETTER ETH (ð)
	tokenMap["\u02a4"] = 82;  // LATIN SMALL LETTER DEZH LIGATURE (ʤ)
	tokenMap["\u0259"] = 83;  // LATIN SMALL LETTER SCHWA (ə)
	tokenMap["\u0258"] = 85;  // LATIN SMALL LETTER R HOOK SCHWA (ɚ)
	tokenMap["\u025b"] = 86;  // LATIN SMALL LETTER OPEN E (ɛ)
	tokenMap["\u025c"] = 87;  // LATIN SMALL LETTER REVERSED OPEN E (ɜ)
	tokenMap["\u025f"] = 90;  // LATIN SMALL LETTER DOTLESS J WITH STROKE (ɟ)
	tokenMap["\u0261"] = 92;  // LATIN SMALL LETTER SCRIPT G (ɡ)
	tokenMap["\u0265"] = 99;  // LATIN SMALL LETTER TURNED H (ɥ)
	tokenMap["\u0268"] = 101; // LATIN SMALL LETTER I WITH STROKE (ɨ)
	tokenMap["\u026a"] = 102; // LATIN SMALL LETTER SMALL CAPITAL I (ɪ)
	tokenMap["\u029d"] = 103; // LATIN SMALL LETTER ESH (ʝ)
	tokenMap["\u026f"] = 110; // LATIN SMALL LETTER TURNED M (ɯ)
	tokenMap["\u0271"] = 111; // LATIN SMALL LETTER M WITH HOOK (ɰ)
	tokenMap["\u0273"] = 112; // LATIN SMALL LETTER ENG (ŋ)
	tokenMap["\u0274"] = 113; // LATIN SMALL LETTER N WITH TAIL (ɳ)
	tokenMap["\u0272"] = 114; // LATIN SMALL LETTER N WITH LEFT HOOK (ɲ)
	tokenMap["\u0274"] = 115; // LATIN SMALL LETTER SMALL CAPITAL N (ɴ)
	tokenMap["\u00f8"] = 116; // LATIN SMALL LETTER O WITH STROKE (ø)
	tokenMap["\u0278"] = 118; // LATIN SMALL LETTER PHI (ɸ)
	tokenMap["\u03b8"] = 119; // GREEK SMALL LETTER THETA (θ)
	tokenMap["\u0153"] = 120; // LATIN SMALL LIGATURE OE (œ)
	tokenMap["\u0279"] = 123; // LATIN SMALL LETTER TURNED R (ɹ)
	tokenMap["\u027e"] = 125; // LATIN SMALL LETTER FLAP R (ɾ)
	tokenMap["\u027b"] = 126; // LATIN SMALL LETTER R WITH HOOK (ɻ)
	tokenMap["\u0281"] = 128; // LATIN SMALL LETTER SMALL CAPITAL R (ʁ)
	tokenMap["\u027d"] = 129; // LATIN SMALL LETTER R WITH FISHHOOK (ɽ)
	tokenMap["\u0282"] = 130; // LATIN SMALL LETTER S WITH CURL (ʂ)
	tokenMap["\u0283"] = 131; // LATIN SMALL LETTER ESH (ʃ)
	tokenMap["\u0288"] = 132; // LATIN SMALL LETTER T WITH RETROFLEX HOOK (ʈ)
	tokenMap["\u02a7"] = 133; // LATIN SMALL LETTER TESH LIGATURE (ʧ)
	tokenMap["\u028a"] = 135; // LATIN SMALL LETTER UPSILON (ʊ)
	tokenMap["\u028b"] = 136; // LATIN SMALL LETTER V WITH HOOK (ʋ)
	tokenMap["\u028c"] = 138; // LATIN SMALL LETTER TURNED V (ʌ)
	tokenMap["\u0263"] = 139; // LATIN SMALL LETTER GAMMA (ɣ)
	tokenMap["\u0264"] = 140; // LATIN SMALL LETTER RAMS HORN (ɤ)
	tokenMap["\u03c7"] = 142; // GREEK SMALL LETTER CHI (χ)
	tokenMap["\u028e"] = 143; // LATIN SMALL LETTER TURNED Y (ʎ)
	tokenMap["\u0292"] = 147; // LATIN SMALL LETTER EZH (ʒ)
	tokenMap["\u0294"] = 148; // LATIN LETTER GLOTTAL STOP (ʔ)
	tokenMap["\u02c8"] = 156; // Primary Stress (ˈ)
	tokenMap["\u02cc"] = 157; // Secondary Stress (ˌ)
	tokenMap["\u02d0"] = 158; // Length Mark (ː)
	tokenMap["\u02b0"] = 162; // Aspirated (ʰ)
	tokenMap["\u02b2"] = 164; // Palatalized (ʲ)
	tokenMap["\u2193"] = 169; // Downstep/Tone (↓)
	tokenMap["\u2192"] = 171; // Tone (→)
	tokenMap["\u2197"] = 172; // Tone (↗)
	tokenMap["\u2198"] = 173; // Tone (↘)
	tokenMap["\u1d7b"] = 177; // Close central unrounded vowel (ᵻ)

	return tokenMap;
}
