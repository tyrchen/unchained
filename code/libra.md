## [aho-corasick (v0.5.3)](https://crates.io/crates/aho-corasick)

A library for finding occurrences of many patterns at once. This library provides multiple pattern search principally through an implementation of the Aho-Corasick algorithm, which builds a fast finite state machine for executing searches in linear time. Features include case insensitive matching, overlapping matches and search & replace in streams.

```rust
extern crate aho_corasick;

use aho_corasick::AhoCorasick;

let patterns = &["apple", "maple", "Snapple"];
let haystack = "Nobody likes maple in their apple flavored Snapple.";

let ac = AhoCorasick::new(patterns);
let mut matches = vec![];
for mat in ac.find_iter(haystack) {
    matches.push((mat.pattern(), mat.start(), mat.end()));
}
assert_eq!(matches, vec![
    (1, 13, 18),
    (0, 28, 33),
    (2, 43, 50),
]);
```

## [arc-swap (v0.3.11)](https://crates.io/crates/arc-swap)

The Rust's Arc can be used from multiple threads and the count is safely updated as needed. However, the Arc itself can't be atomically replaced. To do that, one needs to place it under a lock.

On the other hand, AtomicPtr can be replaced atomically, but it's hard to know when the target can be safely freed.

This is a cross-breed between the two ‒ an ArcSwap can be seeded with an Arc and the Arc can be simultaneously replaced and read by multiple threads.

```rust
extern crate arc_swap;
extern crate crossbeam_utils;

use std::sync::Arc;

use arc_swap::ArcSwap;
use crossbeam_utils::thread;

fn main() {
    let config = ArcSwap::from(Arc::new(String::default()));
    thread::scope(|scope| {
        scope.spawn(|_| {
            let new_conf = Arc::new("New configuration".to_owned());
            config.store(new_conf);
        });
        for _ in 0..10 {
            scope.spawn(|_| {
                loop {
                    let cfg = config.lease();
                    if !cfg.is_empty() {
                        assert_eq!(*cfg, "New configuration");
                        return;
                    }
                }
            });
        }
    }).unwrap();
}
```

## [arrayref (v0.3.5)](https://crates.io/crates/arrayref)

Macros to take array references of slices.

#[macro_use]
extern crate arrayref;

fn read_u16(bytes: &[u8; 2]) -> u16 {
bytes[0] as u16 + ((bytes[1] as u16) << 8)
}
// ...
let data = [0,1,2,3,4,0,6,7,8,9];
assert_eq!(256, read_u16(array_ref![data,0,2]));
assert_eq!(4, read_u16(array_ref![data,4,2]));

```rust
#[macro_use]
extern crate arrayref;

fn read_u16(bytes: &[u8; 2]) -> u16 {
     bytes[0] as u16 + ((bytes[1] as u16) << 8)
}
// ...
let data = [0,1,2,3,4,0,6,7,8,9];
assert_eq!(256, read_u16(array_ref![data,0,2]));
assert_eq!(4, read_u16(array_ref![data,4,2]));
```

## [ascii-canvas (v1.0.0)](https://crates.io/crates/ascii-canvas)

simple canvas for drawing lines and styled text and emitting to the terminal.

## [assert_matches (v1.3.0)](https://crates.io/crates/assert_matches)

Provides a macro, assert_matches, which tests whether a value matches a given pattern, causing a panic if the match fails.

```rust
#[macro_use] extern crate assert_matches;

#[derive(Debug)]
enum Foo {
    A(i32),
    B(i32),
}

let a = Foo::A(1);

assert_matches!(a, Foo::A(_));

assert_matches!(a, Foo::A(i) if i > 0);
```

## [backoff (v0.1.5)](https://crates.io/crates/backoff)

Exponential backoff and retry.

Inspired by the retry mechanism in Google's google-http-java-client library and its Golang port.

```rust
extern crate backoff;
extern crate reqwest;

use backoff::{Error, ExponentialBackoff, Operation};
use reqwest::IntoUrl;

use std::fmt::Display;
use std::io::{self, Read};

fn new_io_err<E: Display>(err: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}

fn fetch_url(url: &str) -> Result<String, Error<io::Error>> {
    let mut op = || {
        println!("Fetching {}", url);
        let url = url.into_url()
            .map_err(new_io_err)
            // Permanent errors need to be explicitly constucted.
            .map_err(Error::Permanent)?;

        let mut resp = reqwest::get(url)
            // Transient errors can be constructed with the ? operator
            // or with the try! macro. No explicit conversion needed
            // from E: Error to backoff::Error;
            .map_err(new_io_err)?;

        let mut content = String::new();
        let _ = resp.read_to_string(&mut content);
        Ok(content)
    };

    let mut backoff = ExponentialBackoff::default();
    op.retry(&mut backoff)
}

fn main() {
    match fetch_url("https::///wrong URL") {
        Ok(_) => println!("Sucessfully fetched"),
        Err(err) => panic!("Failed to fetch: {}", err),
    }
}
```

## [bech32 (v0.6.0)](https://crates.io/crates/bech32)

Rust implementation of the Bech32 encoding format described in BIP-0173.

Bitcoin-specific address encoding is handled by the bitcoin-bech32 crate.

```rust
use bech32::Bech32;

let b = Bech32::new_check_data("bech32".into(), vec![0x00, 0x01, 0x02]).unwrap();
let encoded = b.to_string();
assert_eq!(encoded, "bech321qpz4nc4pe".to_string());

let c = encoded.parse::<Bech32>();
assert_eq!(b, c.unwrap());
```

## [bit-set (v0.5.1)](https://crates.io/crates/bit-set)

An implementation of a set using a bit vector as an underlying representation for holding unsigned numerical elements.

It should also be noted that the amount of storage necessary for holding a set of objects is proportional to the maximum of the objects when viewed as a usize.

```rust
use bit_set::BitSet;

// It's a regular set
let mut s = BitSet::new();
s.insert(0);
s.insert(3);
s.insert(7);

s.remove(7);

if !s.contains(7) {
    println!("There is no 7");
}

// Can initialize from a `BitVec`
let other = BitSet::from_bytes(&[0b11010000]);

s.union_with(&other);

// Print 0, 1, 3 in some order
for x in s.iter() {
    println!("{}", x);
}

// Can convert back to a `BitVec`
let bv = s.into_bit_vec();
assert!(bv[3]);
```

## [bit-vec (v0.5.1)](https://crates.io/crates/bit-vec)

Collections implemented with bit vectors.

```rust
use bit_vec::BitVec;

let max_prime = 10000;

// Store the primes as a BitVec
let primes = {
    // Assume all numbers are prime to begin, and then we
    // cross off non-primes progressively
    let mut bv = BitVec::from_elem(max_prime, true);

    // Neither 0 nor 1 are prime
    bv.set(0, false);
    bv.set(1, false);

    for i in 2.. 1 + (max_prime as f64).sqrt() as usize {
        // if i is a prime
        if bv[i] {
            // Mark all multiples of i as non-prime (any multiples below i * i
            // will have been marked as non-prime previously)
            for j in i.. {
                if i * j >= max_prime {
                    break;
                }
                bv.set(i * j, false)
            }
        }
    }
    bv
};

// Simple primality tests below our max bound
let print_primes = 20;
print!("The primes below {} are: ", print_primes);
for x in 0..print_primes {
    if primes.get(x).unwrap_or(false) {
        print!("{} ", x);
    }
}
println!("");

let num_primes = primes.iter().filter(|x| *x).count();
println!("There are {} primes below {}", num_primes, max_prime);
assert_eq!(num_primes, 1_229);
```

## [bitcoin_hashes (v0.3.2)](https://crates.io/crates/bitcoin_hashes)

This is a simple, no-dependency library which implements the hash functions needed by Bitcoin. These are SHA256, SHA256d, and RIPEMD160. As an ancillary thing, it exposes hexadecimal serialization and deserialization, since these are needed to display hashes anway.

## [blake2 (v0.8.0)](https://crates.io/crates/blake2)

BLAKE2 hash functions.

```rust
use blake2::{Blake2b, Blake2s, Digest};

// create a Blake2b object
let mut hasher = Blake2b::new();

// write input message
hasher.input(b"hello world");

// read hash digest and consume hasher
let res = hasher.result();
assert_eq!(res[..], hex!("
    021ced8799296ceca557832ab941a50b4a11f83478cf141f51f933f653ab9fbc
    c05a037cddbed06e309bf334942c4e58cdf1a46e237911ccd7fcf9787cbc7fd0
")[..]);

// same example for `Blake2s`:
let mut hasher = Blake2s::new();
hasher.input(b"hello world");
let res = hasher.result();
assert_eq!(res[..], hex!("
    9aec6806794561107e594b1f6a8a6b0c92a0cba9acf5e5e93cca06f781813b0b
")[..]);
```

## [block-buffer (v0.7.3)](https://crates.io/crates/block-buffer)

Fixed size buffer for block processing of data.

## [block-padding (v0.1.4)](https://crates.io/crates/block-padding)

Padding and unpadding of messages divided into blocks.

## [bs58 (v0.2.2)](https://crates.io/crates/bs58)

Rust Base58 codec implementation.

Compared to base58 this is significantly faster at decoding (about 2.4x as fast when decoding 32 bytes), almost the same speed for encoding (about 3% slower when encoding 32 bytes), doesn't have the 128 byte limitation and supports a configurable alphabet.

Compared to rust-base58 this is massively faster (over ten times as fast when decoding 32 bytes, almost 40 times as fast when encoding 32 bytes), has no external dependencies and supports a configurable alphabet.

```rust
// basic
let decoded = bs58::decode("he11owor1d").into_vec().unwrap();
let encoded = bs58::encode(decoded).into_string();
assert_eq!("he11owor1d", encoded);

// changing the alphabet
let decoded = bs58::decode("he11owor1d")
    .with_alphabet(bs58::alphabet::RIPPLE)
    .into_vec()
    .unwrap();
let encoded = bs58::encode(decoded)
    .with_alphabet(bs58::alphabet::FLICKR)
    .into_string();
assert_eq!("4DSSNaN1SC", encoded);
```

## [byte-tools (v0.3.1)](https://crates.io/crates/byte-tools)

Bytes related utility functions.

- copy: Copy bytes from src to dst
- set: Sets all bytes in dst equal to value
- zero: Zero all bytes in dst

## [c_linked_list (v1.1.1)](https://crates.io/crates/c_linked_list)

This is a Rust library for handling NULL-terminated C linked lists. You can use this library to take a linked list provided by a C library and wrap it as a Rust type.

```rust
let some_c_linked_list = foreign_function_which_returns_c_linked_list();
let rust_linked_list = unsafe { CLinkedListMut::from_ptr(some_c_linked_list, |n| n.next) };
for (i, node) in rust_linked_list.iter().enumerate() {
    println!("some_c_linked_list[{}] == {}", i, node.value);
}
```

## [chacha20-poly1305-aead (v0.1.2)](https://crates.io/crates/chacha20-poly1305-aead)

This is a pure Rust implementation of the ChaCha20-Poly1305 AEAD from RFC 7539.

## [chashmap (v2.2.2)](https://crates.io/crates/chashmap)

Fast, concurrent hash maps with extensive API.

chashmap is not lockless, but it distributes locks across the map such that lock contentions (which is what could make accesses expensive) are very rare.

Hash maps consists of so called "buckets", which each defines a potential entry in the table. The bucket of some key-value pair is determined by the hash of the key. By holding a read-write lock for each bucket, we ensure that you will generally be able to insert, read, modify, etc. with only one or two locking subroutines.

There is a special-case: reallocation. When the table is filled up such that very few buckets are free (note that this is "very few" and not "no", since the load factor shouldn't get too high as it hurts performance), a global lock is obtained while rehashing the table. This is pretty inefficient, but it rarely happens, and due to the adaptive nature of the capacity, it will only happen a few times when the map has just been initialized.

When two hashes collide, they cannot share the same bucket, so there must be an algorithm which can resolve collisions. In our case, we use linear probing, which means that we take the bucket following it, and repeat until we find a free bucket.

This method is far from ideal, but superior methods like Robin-Hood hashing works poorly (if at all) in a concurrent structure.

```rust

```

## [clear_on_drop (v0.2.3)](https://crates.io/crates/clear_on_drop)

Helpers for clearing sensitive data on the stack and heap.

Some kinds of data should not be kept in memory any longer than they are needed. For instance, cryptographic keys and intermediate values should be erased as soon as they are no longer needed.

The Rust language helps prevent the accidental reading of leftover values on the stack or the heap; however, means outside the program (for instance a debugger, or even physical access to the hardware) can still read the leftover values. For long-lived processes, key material might be found in the memory long after it should have been discarded.

This crate provides two mechanisms to help minimize leftover data.

The ClearOnDrop wrapper holds a mutable reference to sensitive data (for instance, a cipher state), and clears the data when dropped. While the mutable reference is held, the data cannot be moved, so there won't be leftovers due to moves; the wrapper itself can be freely moved. Alternatively, it can hold data on the heap (using a Box<T>, or possibly a similar which allocates from a mlocked heap).

The clear_stack_on_return function calls a closure, and after it returns, overwrites several kilobytes of the stack. This can help overwrite temporary variables used by cryptographic algorithms, and is especially relevant when running on a short-lived thread, since the memory used for the thread stack cannot be easily overwritten after the thread terminates.

```rust
#[derive(Default)]
struct MyData {
    value: Option<u32>,
}

let mut place = MyData { value: Some(0x41414141) };
place.clear();
assert_eq!(place.value, None);

fn as_bytes<T>(x: &T) -> &[u8] {
    unsafe {
        slice::from_raw_parts(x as *const T as *const u8, mem::size_of_val(x))
    }
}
assert!(!as_bytes(&place).contains(&0x41));

```

## [codespan (v0.1.3)](https://crates.io/crates/codespan)

Utilities for dealing with source code locations.

## [codespan-reporting (v0.1.4)](https://crates.io/crates/codespan-reporting)

Diagnostic reporting support for the codespan crate.

To get an idea of what the colored CLI output looks like with codespan-reporting, clone the repository and run the following:

```rust
cargo run -p codespan-reporting --example=emit
cargo run -p codespan-reporting --example=emit -- --color never
```

You should see something like the following in your terminal:

## [cookie (v0.2.5)](https://crates.io/crates/cookie)

Crate for parsing HTTP cookie headers and managing a cookie jar. Supports signed and private (encrypted + signed) jars.

```rust
use cookie::Cookie;

let cookie = Cookie::build("name", "value")
    .domain("www.rust-lang.org")
    .path("/")
    .secure(true)
    .http_only(true)
    .finish();
```

## [crossbeam (v0.4.1)](https://crates.io/crates/crossbeam)

### Atomics

AtomicCell, a thread-safe mutable memory location.(_)
AtomicConsume, for reading from primitive atomic types with "consume" ordering.(_)

### Data structures

- deque, work-stealing deques for building task schedulers.
- ArrayQueue, a bounded MPMC queue that allocates a fixed-capacity buffer on construction.
- SegQueue, an unbounded MPMC queue that allocates small buffers, segments, on demand.

### Memory management

- epoch, an epoch-based garbage collector.(\*\*)

### Thread synchronization

- channel, multi-producer multi-consumer channels for message passing.
- Parker, a thread parking primitive.
- ShardedLock, a sharded reader-writer lock with fast concurrent reads.
- WaitGroup, for synchronizing the beginning or end of some computation.

### Utilities

- Backoff, for exponential backoff in spin loops.(\*)
- CachePadded, for padding and aligning a value to the length of a cache line.(\*)
- scope, for spawning threads that borrow local variables from the stack.

## [crossbeam-channel (v0.2.6)](https://crates.io/crates/crossbeam-channel)

This crate provides multi-producer multi-consumer channels for message passing. It is an alternative to std::sync::mpsc with more features and better performance.

Some highlights:

- Senders and Receivers can be cloned and shared among threads.
- Two main kinds of channels are bounded and unbounded.
- Convenient extra channels like after, never, and tick.
- The select! macro can block on multiple channel operations.
- Select can select over a dynamically built list of channel operations.
- Channels use locks very sparingly for maximum performance.

## [crossbeam-deque (v0.5.2)](https://crates.io/crates/crossbeam-deque)

This crate provides work-stealing deques, which are primarily intended for building task schedulers.

## [crossbeam-epoch (v0.6.1)](https://crates.io/crates/crossbeam-epoch)

This crate provides epoch-based garbage collection for building concurrent data structures.

When a thread removes an object from a concurrent data structure, other threads may be still using pointers to it at the same time, so it cannot be destroyed immediately. Epoch-based GC is an efficient mechanism for deferring destruction of shared objects until no pointers to them can exist.

Everything in this crate except the global GC can be used in no_std + alloc environments.

## [crossbeam-utils (v0.5.0)](https://crates.io/crates/crossbeam-utils)

This crate provides miscellaneous tools for concurrent programming:

### Atomics

- AtomicCell, a thread-safe mutable memory location.(\_)
- AtomicConsume, for reading from primitive atomic types with "consume" ordering.(\_)

### Thread synchronization

- Parker, a thread parking primitive.
- ShardedLock, a sharded reader-writer lock with fast concurrent reads.
- WaitGroup, for synchronizing the beginning or end of some computation.

### Utilities

- Backoff, for exponential backoff in spin loops.(\_)
- CachePadded, for padding and aligning a value to the length of a cache line.(\_)
- scope, for spawning threads that borrow local variables from the stack.
-

Features marked with (\*) can be used in no_std environments.

## [crunchy (v0.1.6)](https://crates.io/crates/crunchy)

Crunchy unroller: deterministically unroll constant loops.

```rust
debug_assert_eq!(MY_CONSTANT, 100);
unroll! {
    for i in 0..100 {
        println!("Iteration {}", i);
    }
}
```

## [crypto-mac (v0.7.0)](https://crates.io/crates/crypto-mac)

Trait for Message Authentication Code (MAC) algorithms.

## [curve25519-dalek (v1.2.1)](https://crates.io/crates/curve25519-dalek)

A pure-Rust implementation of group operations on Ristretto and Curve25519.

curve25519-dalek is a library providing group operations on the Edwards and Montgomery forms of Curve25519, and on the prime-order Ristretto group.

curve25519-dalek is not intended to provide implementations of any particular crypto protocol. Rather, implementations of those protocols (such as x25519-dalek and ed25519-dalek) should use curve25519-dalek as a library.

curve25519-dalek is intended to provide a clean and safe mid-level API for use implementing a wide range of ECC-based crypto protocols, such as key agreement, signatures, anonymous credentials, rangeproofs, and zero-knowledge proof systems.

In particular, curve25519-dalek implements Ristretto, which constructs a prime-order group from a non-prime-order Edwards curve. This provides the speed and safety benefits of Edwards curve arithmetic, without the pitfalls of cofactor-related abstraction mismatches.

## [data-encoding (v2.1.2)](https://crates.io/crates/data-encoding)

This library provides the following common encodings:

- HEXLOWER: lowercase hexadecimal
- HEXLOWER_PERMISSIVE: lowercase hexadecimal with case-insensitive decoding
- HEXUPPER: uppercase hexadecimal
- HEXUPPER_PERMISSIVE: uppercase hexadecimal with case-insensitive decoding
- BASE32: RFC4648 base32
- BASE32_NOPAD: RFC4648 base32 without padding
- BASE32_DNSSEC: RFC5155 base32
- BASE32_DNSCURVE: DNSCurve base32
- BASE32HEX: RFC4648 base32hex
- BASE32HEX_NOPAD: RFC4648 base32hex without padding
- BASE64: RFC4648 base64
- BASE64_NOPAD: RFC4648 base64 without padding
- BASE64_MIME: RFC2045-like base64
- BASE64URL: RFC4648 base64url
- BASE64URL_NOPAD: RFC4648 base64url without padding

```rust
// allocating functions
BASE64.encode(&input_to_encode)
HEXLOWER.decode(&input_to_decode)
// in-place functions
BASE32.encode_mut(&input_to_encode, &mut encoded_output)
BASE64_URL.decode_mut(&input_to_decode, &mut decoded_output)
```

## [datatest (v0.3.1)](https://crates.io/crates/datatest)

Crate for supporting data-driven tests.

Data-driven tests are tests where individual cases are defined via data rather than in code. This crate implements a custom test runner that adds support for additional test types.

```yaml
- name: Pino
  expected: Hi, Pino!
- name: Re-L
  expected: Hi, Re-L!
- name: Vincent
  expected: Hi, Vincent!
```

```rust

#[derive(Deserialize)]
struct GreeterTestCase {
    name: String,
    expected: String,
}

/// Data-driven tests are defined via `#[datatest::data(..)]` attribute.
///
/// This attribute specifies a test file with test cases. Currently, the test file have to be in
/// YAML format. This file is deserialized into `Vec<T>`, where `T` is the type of the test function
/// argument (which must implement `serde::Deserialize`). Then, for each element of the vector, a
/// separate test instance is created and executed.
///
/// Name of each test is derived from the test function module path, test case line number and,
/// optionall, from the [`ToString`] implementation of the test case data (if either [`ToString`]
/// or [`std::fmt::Display`] is implemented).
#[datatest::data("tests/tests.yaml")]
#[test]
fn data_test_line_only(data: &GreeterTestCase) {
    assert_eq!(data.expected, format!("Hi, {}!", data.name));
}
```

## [datatest-derive (v0.3.1)](https://crates.io/crates/datatest-derive)

Procmacro for the datatest crate.

## [derive_deref (v1.1.0)](https://crates.io/crates/derive_deref)

This crate adds a simple `#[derive(Deref)]` and `#[derive(DerefMut)]`. It can be used on any struct with exactly one field. If the type of that field is a reference, the reference will be returned directly.

```rust
#[derive(Deref)]
struct MyInt(i32);

assert_eq!(&1, &*MyInt(1));
assert_eq!(&2, &*MyInt(2));

#[derive(Deref)]
struct MyString<'a>(&'a str);

// Note that we deref to &str, not &&str
assert_eq!("foo", &*MyString("foo"));
assert_eq!("bar", &*MyString("bar"));
```

## [diff (v0.1.11)](https://crates.io/crates/diff)

An LCS based slice and string diffing implementation.

```rust
extern crate diff;

fn main() {
    let left = "foo\nbar\nbaz\nquux";
    let right = "foo\nbaz\nbar\nquux";

    for diff in diff::lines(left, right) {
        match diff {
            diff::Result::Left(l)    => println!("-{}", l),
            diff::Result::Both(l, _) => println!(" {}", l),
            diff::Result::Right(r)   => println!("+{}", r)
        }
    }
}
```

outputs:

```diff
 foo
-bar
 baz
+bar
 quux
```

## [digest (v0.8.0)](https://crates.io/crates/digest)

Traits for cryptographic hash functions.

This crate provides traits which describe functionality of cryptographic hash functions.

Traits in this repository can be separated in two levels:

Low level traits: Input, BlockInput, Reset, FixedOutput, VariableOutput, ExtendableOutput. These traits atomically describe available functionality of hash function implementations.
Convenience trait: Digest, DynDigest. They are wrappers around low level traits for most common hash-function use-cases.
Additionally hash functions implement traits from std: Default, Clone, Write. (the latter depends on enabled-by-default std crate feature)

The Digest trait is the most commonly used trait.

```rust
println!("{:x}", sha2::Sha256::digest(b"Hello world"));
```

## [docopt (v1.1.0)](https://crates.io/crates/docopt)

Docopt for Rust with automatic type based decoding (i.e., data validation). This implementation conforms to the official description of Docopt and passes its test suite.

```rust
use docopt::Docopt;
use serde::Deserialize;

const USAGE: &'static str = "
Naval Fate.

Usage:
  naval_fate.py ship new <name>...
  naval_fate.py ship <name> move <x> <y> [--speed=<kn>]
  naval_fate.py ship shoot <x> <y>
  naval_fate.py mine (set|remove) <x> <y> [--moored | --drifting]
  naval_fate.py (-h | --help)
  naval_fate.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --speed=<kn>  Speed in knots [default: 10].
  --moored      Moored (anchored) mine.
  --drifting    Drifting mine.
";

#[derive(Debug, Deserialize)]
struct Args {
    flag_speed: isize,
    flag_drifting: bool,
    arg_name: Vec<String>,
    arg_x: Option<i32>,
    arg_y: Option<i32>,
    cmd_ship: bool,
    cmd_mine: bool,
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    println!("{:?}", args);
}
```

## [ed25519-dalek (v1.0.0-pre.1)](https://crates.io/crates/ed25519-dalek)

Fast and efficient Rust implementation of ed25519 key generation, signing, and verification in Rust.

```rust
extern crate rand;
extern crate ed25519_dalek;

use rand::Rng;
use rand::rngs::OsRng;
use ed25519_dalek::Keypair;
use ed25519_dalek::Signature;

let mut csprng: OsRng = OsRng::new().unwrap();
let keypair: Keypair = Keypair::generate(&mut csprng);

let message: &[u8] = b"This is a test of the tsunami alert system.";
let signature: Signature = keypair.sign(message);

assert!(keypair.verify(message, &signature).is_ok());

```

anyone else with public key could verify:

```rust
use ed25519_dalek::PublicKey;

let public_key: PublicKey = keypair.public;
assert!(public_key.verify(message, &signature).is_ok());
```

## [ena (v0.11.0)](https://crates.io/crates/ena)

An implementation of union-find in Rust; extracted from (and used by) rustc.

## [endian-type (v0.1.2)](https://crates.io/crates/endian-type)

Type safe wrappers for types with a defined byte order.

## [errno (v0.2.4)](https://crates.io/crates/errno)

Cross-platform interface to the `errno` variable.

```rust
extern crate errno;
use errno::{Errno, errno, set_errno};

// Get the current value of errno
let e = errno();

// Set the current value of errno
set_errno(e);

// Extract the error code as an i32
let code = e.0;

// Display a human-friendly error message
println!("Error {}: {}", code, e);
```

## [fake-simd (v0.1.2)](https://crates.io/crates/fake-simd)

Crate for mimicking simd crate on stable Rust.

## [filecheck (v0.4.0)](https://crates.io/crates/filecheck)

This is a library for writing tests for utilities that read text files and produce text output.

```rust
fn is_prime(x: u32) -> bool {
    (2..x).all(|d| x % d != 0)
}

// Check that we get the primes and nothing else:
//   regex: NUM=\d+
//   not: $NUM
//   check: 2
//   nextln: 3
//   check: 89
//   nextln: 97
//   not: $NUM
fn main() {
    for p in (2..10).filter(|&x| is_prime(x)) {
        println!("{}", p);
    }
}
```

## [fs_extra (v1.1.0)](https://crates.io/crates/fs_extra)

Expanding opportunities standard library std::fs and std::io. Recursively copy folders with recept information about process and much more.

```rust
use std::path::Path;
use std::{thread, time};
use std::sync::mpsc::{self, TryRecvError};

extern crate fs_extra;
use fs_extra::dir::*;
use fs_extra::error::*;

fn example_copy() -> Result<()> {

    let path_from = Path::new("./temp");
    let path_to = path_from.join("out");
    let test_folder = path_from.join("test_folder");
    let dir = test_folder.join("dir");
    let sub = dir.join("sub");
    let file1 = dir.join("file1.txt");
    let file2 = sub.join("file2.txt");

    create_all(&sub, true)?;
    create_all(&path_to, true)?;
    fs_extra::file::write_all(&file1, "content1")?;
    fs_extra::file::write_all(&file2, "content2")?;

    assert!(dir.exists());
    assert!(sub.exists());
    assert!(file1.exists());
    assert!(file2.exists());


    let mut options = CopyOptions::new();
    options.buffer_size = 1;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let handler = |process_info: TransitProcess| {
            tx.send(process_info).unwrap();
            thread::sleep(time::Duration::from_millis(500));
            fs_extra::dir::TransitProcessResult::ContinueOrAbort
        };
        copy_with_progress(&test_folder, &path_to, &options, handler).unwrap();
    });

    loop {
        match rx.try_recv() {
            Ok(process_info) => {
                println!("{} of {} bytes",
                         process_info.copied_bytes,
                         process_info.total_bytes);
            }
            Err(TryRecvError::Disconnected) => {
                println!("finished");
                break;
            }
            Err(TryRecvError::Empty) => {}
        }
    }
    Ok(())

}
fn main() {
    example_copy();
}
```

## [futures-channel-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-channel-preview)

Channels for asynchronous communication using futures-rs.

## [futures-core-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-core-preview)

The core traits and types in for the `futures` library.

## [futures-executor-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-executor-preview)

Executors for asynchronous tasks based on the futures-rs library.

## [futures-io-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-io-preview)

The `AsyncRead` and `AsyncWrite` traits for the futures-rs library.

## [futures-locks (v0.3.0)](https://crates.io/crates/futures-locks)

A library of Futures-aware locking primitives. These locks can safely be used in asynchronous environments like Tokio. When they block, they'll only block a single task, not the entire reactor.

## [futures-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-preview)

This library is an implementation of zero-cost futures in Rust.

## [futures-select-macro-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-select-macro-preview)

The `select!` macro for waiting on multiple different `Future`s at once and handling the first one to complete.

## [futures-sink-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-sink-preview)

The asynchronous `Sink` trait for the futures-rs library.

## [futures-util-preview (v0.3.0-alpha.16)](https://crates.io/crates/futures-util-preview)

Common utilities and extension traits for the futures-rs library.

## [generic-array (v0.12.0)](https://crates.io/crates/generic-array)

This crate implements generic array types for Rust.

generic-array defines a new trait `ArrayLength<T>` and a struct `GenericArray<T, N: ArrayLength<T>>`.

```rust
struct Foo<N: ArrayLength<i32>> {
	data: GenericArray<i32, N>
}

let array = arr![u32; 1, 2, 3];
assert_eq!(array[2], 3);
```

## [get_if_addrs (v0.5.3)](https://crates.io/crates/get_if_addrs)

Retrieve network interface info for all interfaces on the system.

```rust
// List all of the machine's network interfaces
for ifa in get_if_addrs::get_if_addrs().unwrap() {
    println!("{:#?}", ifa);
}
```

## [getopts (v0.2.19)](https://crates.io/crates/getopts)

A Rust library for option parsing for CLI utilities.

## [getrandom (v0.1.3)](https://crates.io/crates/getrandom)

A small cross-platform library for retrieving random data from system source.

## [grpcio (v0.4.4)](https://crates.io/crates/grpcio)

gRPC-rs is a Rust wrapper of gRPC Core. gRPC is a high performance, open source universal RPC framework that puts mobile and HTTP/2 first.

## [grpcio-compiler (v0.4.3)](https://crates.io/crates/grpcio-compiler)

gRPC compiler for grpcio.

## [grpcio-sys (v0.4.4)](https://crates.io/crates/grpcio-sys)

FFI bindings to gRPC c core library.

## [h2 (v0.1.24)](https://crates.io/crates/h2)

A Tokio aware, HTTP/2.0 client & server implementation for Rust.

- Client and server HTTP/2.0 implementation.
- Implements the full HTTP/2.0 specification.
- Passes h2spec.
- Focus on performance and correctness.
- Built on Tokio.

## [hex (v0.3.2)](https://crates.io/crates/hex)

Encoding and decoding data into/from hexadecimal representation.

```rust
extern crate hex;

fn main() {
    let hex_string = hex::encode("Hello world!");
    println!("{}", hex_string); // Prints '48656c6c6f20776f726c6421'
}
```

## [hex_fmt (v0.3.0)](https://crates.io/crates/hex_fmt)

Formatting and shortening byte slices as hexadecimal strings.

This crate provides wrappers for byte slices and lists of byte slices that implement the standard formatting traits and print the bytes as a hexadecimal string. It respects the alignment, width and precision parameters and applies padding and shortening.

```rust
let bytes: &[u8] = &[0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f];

assert_eq!("0a1b2c3d4e5f", &format!("{}", HexFmt(bytes)));

// By default the full slice is printed. Change the width to apply padding or shortening.
assert_eq!("0a..5f", &format!("{:6}", HexFmt(bytes)));
assert_eq!("0a1b2c3d4e5f", &format!("{:12}", HexFmt(bytes)));
assert_eq!("  0a1b2c3d4e5f  ", &format!("{:16}", HexFmt(bytes)));

// The default alignment is centered. Use `<` or `>` to align left or right.
assert_eq!("0a1b..", &format!("{:<6}", HexFmt(bytes)));
assert_eq!("0a1b2c3d4e5f    ", &format!("{:<16}", HexFmt(bytes)));
assert_eq!("..4e5f", &format!("{:>6}", HexFmt(bytes)));
assert_eq!("    0a1b2c3d4e5f", &format!("{:>16}", HexFmt(bytes)));

// Use e.g. `4.8` to set the minimum width to 4 and the maximum to 8.
assert_eq!(" 12 ", &format!("{:4.8}", HexFmt([0x12])));
assert_eq!("123456", &format!("{:4.8}", HexFmt([0x12, 0x34, 0x56])));
assert_eq!("123..89a", &format!("{:4.8}", HexFmt([0x12, 0x34, 0x56, 0x78, 0x9a])));

// If you prefer uppercase, use `X`.
assert_eq!("0A1B..4E5F", &format!("{:X}", HexFmt(bytes)));

// All of the above can be combined.
assert_eq!("0A1B2C..", &format!("{:<4.8X}", HexFmt(bytes)));

// With `HexList`, the parameters are applied to each entry.
let list = &[[0x0a; 3], [0x1b; 3], [0x2c; 3]];
assert_eq!("[0A.., 1B.., 2C..]", &format!("{:<4X}", HexList(list)));
```

## [hmac (v0.7.0)](https://crates.io/crates/hmac)

Generic implementation of Hash-based Message Authentication Code (HMAC).

## [hpack (v0.2.0)](https://crates.io/crates/hpack)

An HPACK coder implementation in Rust.

The library lets you perform header compression and decompression according to the HPACK spec.

The decoder module implements the API for performing HPACK decoding. The Decoder struct will track the decoding context during its lifetime (i.e. subsequent headers on the same connection should be decoded using the same instance).

The decoder implements the full spec and allows for decoding any valid sequence of bytes representing a compressed header list.

The encoder module implements the API for performing HPACK encoding. The Encoder struct will track the encoding context during its lifetime (i.e. the same instance should be used to encode all headers on the same connection).

The encoder so far does not implement Huffman string literal encoding; this, however, is enough to be able to send requests to any HPACK-compliant server, as Huffman encoding is completely optional.

Encoding:

```rust
se hpack::Encoder;

let mut encoder = Encoder::new();
let headers = vec![
    (b":method".to_vec(), b"GET".to_vec()),
    (b":path".to_vec(), b"/".to_vec()),
];
// The headers are encoded by providing their index (with a bit flag
// indicating that the indexed representation is used).
assert_eq!(encoder.encode(&headers), vec![2 | 0x80, 4 | 0x80]);
```

Decoding:

```rust
use hpack::Decoder;

let mut decoder = Decoder::new();
let header_list = decoder.decode(&[0x82, 0x84]).unwrap();
assert_eq!(header_list, [
    (b":method".to_vec(), b"GET".to_vec()),
    (b":path".to_vec(), b"/".to_vec()),
]);
```

## [hyper (v0.12.30)](https://crates.io/crates/hyper)

A fast and correct HTTP implementation for Rust.

hyper is a fast, safe HTTP implementation written in and for Rust.

hyper offers both an HTTP client and server which can be used to drive complex web applications written entirely in Rust.

hyper makes use of "async IO" (non-blocking sockets) via the Tokio and Futures crates.

Be aware that hyper is still actively evolving towards 1.0, and is likely to experience breaking changes before stabilising. However, this mostly now around the instability of Future and async. The rest of the API is rather stable now. You can also see the issues in the upcoming milestones.

## [itertools (v0.7.11)](https://crates.io/crates/itertools)

Extra iterator adaptors, iterator methods, free functions, and macros.

```rust
use itertools::Itertools;

let it = (1..3).interleave(vec![-1, -2]);
itertools::assert_equal(it, vec![1, -1, 2, -2]);

```

## [jemalloc-sys (v0.1.8)](https://crates.io/crates/jemalloc-sys)

jemalloc-sys - Rust bindings to the jemalloc C library.

## [jemallocator (v0.1.9)](https://crates.io/crates/jemallocator)

Links against jemalloc and provides a Jemalloc unit type that implements the allocator APIs and can be set as the `#[global_allocator]`.

## [keccak (v0.1.0)](https://crates.io/crates/keccak)

Keccak-f sponge function

## [lalrpop (v0.16.3)](https://crates.io/crates/lalrpop)

LALRPOP is a Rust parser generator framework with usability as its primary goal. You should be able to write compact, DRY, readable grammars. To this end, LALRPOP offers a number of nifty features:

- Nice error messages in case parser constructor fails.
- Macros that let you extract common parts of your grammar. This means you can go beyond simple repetition like Id\* and define things like Comma<Id> for a comma-separated list of identifiers.
- Macros can also create subsets, so that you easily do something like Expr<"all"> to represent the full range of expressions, but Expr<"if"> to represent the subset of expressions that can appear in an if expression.
- Builtin support for operators like \* and ?.
- Compact defaults so that you can avoid writing action code much of the time.
- Type inference so you can often omit the types of nonterminals.

Despite its name, LALRPOP in fact uses LR(1) by default (though you can opt for LALR(1)), and really I hope to eventually move to something general that can handle all CFGs (like GLL, GLR, LL(\*), etc).

## [lalrpop-util (v0.16.3)](https://crates.io/crates/lalrpop-util)

Runtime library for parsers generated by LALRPOP.

## [language-tags (v0.2.2)](https://crates.io/crates/language-tags)

Language tags for Rust.

```rust
use language_tags::LanguageTag;
use std::iter::FromIterator;
let langtag: LanguageTag = "en-x-twain".parse().unwrap();
assert_eq!(langtag.primary_language(), "en");
assert_eq!(Vec::from_iter(langtag.private_use_subtags()), vec!["twain"]);
```

## [lru-cache (v0.1.2)](https://crates.io/crates/lru-cache)

A cache that holds a limited number of key-value pairs.

```rust
use lru_cache::LruCache;

let mut cache = LruCache::new(2);

cache.insert(1, 10);
cache.insert(2, 20);
cache.insert(3, 30);
assert!(cache.get_mut(&1).is_none());
assert_eq!(*cache.get_mut(&2).unwrap(), 20);
assert_eq!(*cache.get_mut(&3).unwrap(), 30);

cache.insert(2, 22);
assert_eq!(*cache.get_mut(&2).unwrap(), 22);

cache.insert(6, 60);
assert!(cache.get_mut(&3).is_none());

cache.set_capacity(1);
assert!(cache.get_mut(&2).is_none());
```

## [mach_o_sys (v0.1.1)](https://crates.io/crates/mach_o_sys)

Bindings to the OSX mach-o system library.

## [memchr (v0.1.11)](https://crates.io/crates/memchr)

The memchr crate provides heavily optimized routines for searching bytes.

```rust
use memchr::memchr;

let haystack = b"the quick brown fox";
assert_eq!(memchr(b'k', haystack), Some(8));
```

## [memsec (v0.5.6)](https://crates.io/crates/memsec)

Rust implementation `libsodium/utils`.

## [mime (v0.2.6)](https://crates.io/crates/mime)

Strongly Typed Mimes.

## [mirai-annotations (v0.1.0)](https://crates.io/crates/mirai-annotations)

This crate provides a set of macros that can be used in the place of the standard RUST assert and debug_assert macros. They add value by allowing MIRAI to:

- distinguish between path conditions and verification conditions
- distinguish between conditions that it should assume as true and conditions that it should verify
- check conditions at compile time that should not be checked at runtime because they are too expensive

From this it follows that we have three families of macros:

- assume macros
- precondition macros (like assume where defined and like verify for callers)
- verify macros

Each of these has three kinds

- only checked at compile time ('macro' with macro among {assume, precondition, verify})
- always checked at runtime ('checked_macro')
- checked at runtime only for debug builds ('debug_checked_macro')

Additionally, the runtime checked kinds provides eq and ne varieties, leaving us with:

- assume!
- checked_assume!
- checked_assume_eq!
- checked_assume_ne!
- debug_checked_assume!
- debug_checked_assume_eq!
- debug_checked_assume_ne!
- precondition!
- checked_precondition!
- checked_precondition_eq!
- checked_precondition_ne!
- debug_checked_precondition!
- debug_checked_precondition_eq!
- debug_checked_precondition_ne!
- verify!
- checked_verify!
- checked_verify_eq!
- checked_verify_ne!
- debug_checked_verify!
- debug_checked_verify_eq!
- debug_checked_verify_ne!

This crate also provides macros for describing and constraining abstract state that only has meaning to MIRAI. These are:

- get_model_field!
- result!
- set_model_field!

See the documentation for details on how to use these.

## [mktemp (v0.3.1)](https://crates.io/crates/mktemp)

mktemp files and directories。

## [nibble_vec (v0.0.4)](https://crates.io/crates/nibble_vec)

Data-structure for storing a sequence of half-bytes.

Wraps a `Vec<u8>`, providing safe and memory-efficient storage of 4-bit values.

In terms of supported operations, the structure behaves kind of like a fixed length array, in that insertions into the middle of the vector are difficult (and unimplemented at present).

## [nix (v0.13.1)](https://crates.io/crates/nix)

Rust friendly bindings to \*nix APIs.

## [nohash-hasher (v0.1.1)](https://crates.io/crates/nohash-hasher)

A `NoHashHasher<T>` where T is one of `{u8, u16, u32, u64, usize, i8, i16, i32, i64, isize}` is a stateless implementation of std::hash::Hasher which does not actually hash at all.

By itself this hasher is largely useless, but when used in HashMaps whose domain matches T the resulting map operations involving hashing are faster than with any other possible hashing algorithm.

Using this hasher, one must ensure that it is never used in a stateful way, i.e. a single call to `write_*` must be followed by `finish`. Multiple write-calls will cause errors (debug builds check this and panic if a violation of this API contract is detected).

## [opaque-debug (v0.2.2)](https://crates.io/crates/opaque-debug)

Macro for opaque Debug trait implementation.

## [owning_ref (v0.3.3)](https://crates.io/crates/owning_ref)

A library for creating references that carry their owner with them.

```rust
fn return_owned_and_referenced() -> OwningRef<Vec<u8>, [u8]> {
    let v = vec![1, 2, 3, 4];
    let or = OwningRef::new(v);
    let or = or.map(|v| &v[1..3]);
    or
}
```

## [pairing (v0.14.2)](https://crates.io/crates/pairing)

Pairing-friendly elliptic curve library.

## [parity-multiaddr (v0.4.1)](https://crates.io/crates/parity-multiaddr)

Implementation of the multiaddr format.

## [parity-multihash (v0.1.2)](https://crates.io/crates/parity-multihash)

Implementation of the multihash format.

## [parking_lot (v0.6.4)](https://crates.io/crates/parking_lot)

This library provides implementations of `Mutex`, `RwLock`, `Condvar` and `Once` that are smaller, faster and more flexible than those in the Rust standard library, as well as a ReentrantMutex type which supports recursive locking. It also exposes a low-level API for creating your own efficient synchronization primitives.

When tested on x86_64 Linux, parking_lot::Mutex was found to be 1.5x faster than std::sync::Mutex when uncontended, and up to 5x faster when contended from multiple threads. The numbers for RwLock vary depending on the number of reader and writer threads, but are almost always faster than the standard library RwLock, and even up to 50x faster in some cases.

## [parking_lot_core (v0.3.1)](https://crates.io/crates/parking_lot_core)

An advanced API for creating custom synchronization primitives.

## [pin-utils (v0.1.0-alpha.4)](https://crates.io/crates/pin-utils)

Utilities for pinning.

## [proc-macro-hack (v0.5.7)](https://crates.io/crates/proc-macro-hack)

As of Rust 1.30, the language supports user-defined function-like procedural macros. However these can only be invoked in item position, not in statements or expressions.

This crate implements an alternative type of procedural macro that can be invoked in statement or expression position.

This approach works with any stable or nightly Rust version 1.30+.

## [proc-macro-nested (v0.1.3)](https://crates.io/crates/proc-macro-nested)

Support for nested proc-macro-hack invocations.

## [prometheus (v0.4.2)](https://crates.io/crates/prometheus)

This is the Rust client library for Prometheus. The main Structures and APIs are ported from Go client.

```rust
use prometheus::{Opts, Registry, Counter, TextEncoder, Encoder};

// Create a Counter.
let counter_opts = Opts::new("test_counter", "test counter help");
let counter = Counter::with_opts(counter_opts).unwrap();

// Create a Registry and register Counter.
let r = Registry::new();
r.register(Box::new(counter.clone())).unwrap();

// Inc.
counter.inc();

// Gather the metrics.
let mut buffer = vec![];
let encoder = TextEncoder::new();
let metric_families = r.gather();
encoder.encode(&metric_families, &mut buffer).unwrap();

// Output to the standard output.
println!("{}", String::from_utf8(buffer).unwrap());
```

## [proptest (v0.9.4)](https://crates.io/crates/proptest)

Proptest is a property testing framework (i.e., the QuickCheck family) inspired by the Hypothesis framework for Python. It allows to test that certain properties of your code hold for arbitrary inputs, and if a failure is found, automatically finds the minimal test case to reproduce the problem. Unlike QuickCheck, generation and shrinking is defined on a per-value basis instead of per-type, which makes it more flexible and simplifies composition.

```rust
// Bring the macros and other important things into scope.
use proptest::prelude::*;

fn parse_date(s: &str) -> Option<(u32, u32, u32)> {
    if 10 != s.len() { return None; }
    if "-" != &s[4..5] || "-" != &s[7..8] { return None; }

    let year = &s[0..4];
    let month = &s[6..7];
    let day = &s[8..10];

    year.parse::<u32>().ok().and_then(
        |y| month.parse::<u32>().ok().and_then(
            |m| day.parse::<u32>().ok().map(
                |d| (y, m, d))))
}
proptest! {
    #[test]
    fn doesnt_crash(s in "\\PC*") {
        parse_date(&s);
    }
}
```

## [proptest-derive (v0.1.2)](https://crates.io/crates/proptest-derive)

Custom-derive for the Arbitrary trait of proptest.

## [protoc (v2.6.2)](https://crates.io/crates/protoc)

Protobuf protoc command as API.

## [protoc-grpcio (v0.3.1)](https://crates.io/crates/protoc-grpcio)

API for programatically invoking the grpcio (grpc-rs) gRPC compiler.

## [protoc-rust (v2.6.2)](https://crates.io/crates/protoc-rust)

protoc --rust_out=... available as API. protoc needs to be in \$PATH, protoc-gen-run does not.

## [quick-error (v0.2.2)](https://crates.io/crates/quick-error)

A macro which makes error types pleasant to write.

```rust
quick_error! {
    #[derive(Debug)]
    pub enum IoWrapper {
        Io(err: io::Error) {
            from()
            description("io error")
            display("I/O error: {}", err)
            cause(err)
        }
        Other(descr: &'static str) {
            description(descr)
            display("Error {}", descr)
        }
        IoAt { place: &'static str, err: io::Error } {
            cause(err)
            display(me) -> ("{} {}: {}", me.description(), place, err)
            description("io error at")
            from(s: String) -> {
                place: "some string",
                err: io::Error::new(io::ErrorKind::Other, s)
            }
        }
        Discard {
            from(&'static str)
        }
    }
}
```

## [radix_trie (v0.1.4)](https://crates.io/crates/radix_trie)

This is a Radix Trie implementation in Rust, building on the lessons learnt from TrieMap and Sequence Trie. You can read about my experience implementing this data structure here.

## [rand (v0.5.6)](https://crates.io/crates/rand)

A Rust library for random number generation.

Rand provides utilities to generate random numbers, to convert them to useful types and distributions, and some randomness-related algorithms.

The core random number generation traits of Rand live in the rand_core crate but are also exposed here; RNG implementations should prefer to use rand_core while most other users should depend on rand.

## [rand04 (v0.1.1)](https://crates.io/crates/rand04)

This can be used instead of the deprecated rand 0.4 in crates that for reasons of compatibility need to depend on both a current version and 0.4.

## [rand04_compat (v0.1.1)](https://crates.io/crates/rand04_compat)

Wrappers for the deprecated rand 0.4 that allow generating values implementing the 0.4 Rand trait with RNGs from rand 0.6.

## [regex (v0.1.80)](https://crates.io/crates/regex)

A Rust library for parsing, compiling, and executing regular expressions. Its syntax is similar to Perl-style regular expressions, but lacks a few features like look around and backreferences. In exchange, all searches execute in linear time with respect to the size of the regular expression and search text. Much of the syntax and implementation is inspired by RE2.

```rust
use regex::Regex;

const TO_SEARCH: &'static str = "
On 2010-03-14, foo happened. On 2014-10-14, bar happened.
";

fn main() {
    let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();

    for caps in re.captures_iter(TO_SEARCH) {
        // Note that all of the unwraps are actually OK for this regex
        // because the only way for the regex to match is if all of the
        // capture groups match. This is not true in general though!
        println!("year: {}, month: {}, day: {}",
                 caps.get(1).unwrap().as_str(),
                 caps.get(2).unwrap().as_str(),
                 caps.get(3).unwrap().as_str());
    }
}
```

## [regex-syntax (v0.3.9)](https://crates.io/crates/regex-syntax)

A regular expression parser.

## [rental (v0.5.4)](https://crates.io/crates/rental)

Rental - A macro to generate safe self-referential structs, plus premade types for common use cases.

It can sometimes occur in the course of designing an API that it would be convenient, or even necessary, to allow fields within a struct to hold references to other fields within that same struct. Rust's concept of ownership and borrowing is powerful, but can't express such a scenario yet. //! Creating such a struct manually would require unsafe code to erase lifetime parameters from the field types. Accessing the fields directly would be completely unsafe as a result. This library addresses that issue by allowing access to the internal fields only under carefully controlled circumstances, through closures that are bounded by generic lifetimes to prevent infiltration or exfiltration of any data with an incorrect lifetime. In short, while the struct internally uses unsafe code to store the fields, the interface exposed to the consumer of the struct is completely safe. The implementation of this interface is subtle and verbose, hence the macro to automate the process.

The API of this crate consists of the rental macro that generates safe self-referential structs, a few example instantiations to demonstrate the API provided by such structs (see examples), and a module of premade instantiations to cover common use cases (see common).

```rust
rental! {
    pub mod rent_libloading {
        use libloading;

        #[rental(deref_suffix)] // This struct will deref to the Deref::Target of Symbol.
        pub struct RentSymbol<S: 'static> {
            lib: Box<libloading::Library>, // Library is boxed for StableDeref.
            sym: libloading::Symbol<'lib, S>, // The 'lib lifetime borrows lib.
        }
    }
}

fn main() {
    let lib = libloading::Library::new("my_lib.so").unwrap(); // Open our dylib.
    if let Ok(rs) = rent_libloading::RentSymbol::try_new(
        Box::new(lib),
        |lib| unsafe { lib.get::<extern "C" fn()>(b"my_symbol") }) // Loading symbols is unsafe.
    {
        (*rs)(); // Call our function
    };
}
```

## [rental-impl (v0.5.4)](https://crates.io/crates/rental-impl)

An implementation detail of rental. Should not be used directly.

## [ring (v0.14.6)](https://crates.io/crates/ring)

ring is focused on general-purpose cryptography. WebPKI X.509 certificate validation is done in the webpki project, which is built on top of ring. Also, multiple groups are working on implementations of cryptographic protocols like TLS, SSH, and DNSSEC on top of ring.

## [rmp (v0.8.7)](https://crates.io/crates/rmp)

RMP is a pure Rust MessagePack implementation of an efficient binary serialization format. This crate provides low-level core functionality, writers and readers for primitive values with direct mapping between binary MessagePack format.

## [rmp-serde (v0.13.7)](https://crates.io/crates/rmp-serde)

This repository consists of three separate crates: the RMP core and two implementations to ease serializing and deserializing Rust structs.

## [rust_decimal (v1.0.1)](https://crates.io/crates/rust_decimal)

A Decimal implementation written in pure Rust suitable for financial calculations that require significant integral and fractional digits with no round-off errors.

The binary representation consists of a 96 bit integer number, a scaling factor used to specify the decimal fraction and a 1 bit sign. Because of this representation, trailing zeros are preserved and may be exposed when in string form. These can be truncated using the normalize or round_dp functions.

```rust
use rust_decimal_macros::*;

let number = dec!(-1.23);

use rust_decimal::Decimal;

// Using an integer followed by the decimal points
let scaled = Decimal::new(202, 2); // 2.02

// From a string representation
let from_string = Decimal::from_str("2.02").unwrap(); // 2.02

// Using the `Into` trait
let my_int : Decimal = 3i32.into();

// Using the raw decimal representation
// 3.1415926535897932384626433832
let pi = Decimal::from_parts(1102470952, 185874565, 1703060790, false, 28);
```

## [rust-crypto (v0.2.36)](https://crates.io/crates/rust-crypto)

A (mostly) pure-Rust implementation of various common cryptographic algorithms.

Rust-Crypto seeks to create practical, auditable, pure-Rust implementations of common cryptographic algorithms with a minimum amount of assembly code where appropriate. The x86-64, x86, and ARM architectures are supported, although the x86-64 architecture receives the most testing.

Rust-Crypto targets the current, stable build of Rust. If you are having issues while using an older version, please try upgrading to the latest stable.

Rust-Crypto has not been thoroughly audited for correctness, so any use where security is important is not recommended at this time.

- AES
- Bcrypt
- Blake2B
- Blowfish
- ChaCha20
- Curve25519
- ECB, CBC, and CTR block cipher modes
- Ed25519
- Fortuna
- Ghash
- HC128
- HMAC
- MD5
- PBKDF2
- PKCS padding for CBC block cipher mode
- Poly1305
- RC4
- RIPEMD-160
- Salsa20 and XSalsa20
- Scrypt
- Sha1
- Sha2 (All fixed output size variants)
- Sosemanuk
- Whirlpool

## [rusty-fork (v0.2.2)](https://crates.io/crates/rusty-fork)

Rusty-fork provides a way to "fork" unit tests into separate processes.

There are a number of reasons to want to run some tests in isolated processes:

- When tests share a process, if any test causes the process to abort, segfault, overflow the stack, etc., the entire test runner process dies. If the test is in a subprocess, only the subprocess dies and the test runner simply fails the test.
- Isolating a test to a subprocess makes it possible to add a timeout to the test and forcibly terminate it and produce a normal test failure.
- Tests which need to interact with some inherently global property, such as the current working directory, can do so without interfering with other tests.

This crate itself provides two things:

- The rusty_fork_test! macro, which is a simple way to wrap standard Rust tests to be run in subprocesses with optional timeouts.
- The fork function which can be used as a building block to make other types of process isolation strategies.

## [rustyline (v4.1.0)](https://crates.io/crates/rustyline)

Readline implementation in Rust that is based on Antirez' Linenoise.

```rust
extern crate rustyline;

use rustyline::error::ReadlineError;
use rustyline::Editor;

fn main() {
    // `()` can be used when no completer is required
    let mut rl = Editor::<()>::new();
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_ref());
                println!("Line: {}", line);
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break
            },
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break
            }
        }
    }
    rl.save_history("history.txt").unwrap();
}
```

## [serde_yaml (v0.8.9)](https://crates.io/crates/serde_yaml)

This crate is a Rust library for using the Serde serialization framework with data in YAML file format.

```rust
use std::collections::BTreeMap;

fn main() -> Result<(), serde_yaml::Error> {
    // You have some type.
    let mut map = BTreeMap::new();
    map.insert("x".to_string(), 1.0);
    map.insert("y".to_string(), 2.0);

    // Serialize it to a YAML string.
    let s = serde_yaml::to_string(&map)?;
    assert_eq!(s, "---\nx: 1\ny: 2");

    // Deserialize it back to a Rust type.
    let deserialized_map: BTreeMap<String, f64> = serde_yaml::from_str(&s)?;
    assert_eq!(map, deserialized_map);
    Ok(())
}
```

## [sha-1 (v0.8.1)](https://crates.io/crates/sha-1)

SHA-1 hash function

## [sha2 (v0.8.0)](https://crates.io/crates/sha2)

SHA-2 hash functions

## [sha3 (v0.8.2)](https://crates.io/crates/sha3)

SHA-3 (Keccak) hash function

## [signal-hook (v0.1.9)](https://crates.io/crates/signal-hook)

Library for safe and correct Unix signal handling in Rust.

Unix signals are inherently hard to handle correctly, for several reasons:

- They are a global resource. If a library wants to set its own signal handlers, it risks disturbing some other library. It is possible to chain the previous signal handler, but then it is impossible to remove the old signal handlers from the chains in any practical manner.
- They can be called from whatever thread, requiring synchronization. Also, as they can interrupt a thread at any time, making most handling race-prone.
- According to the POSIX standard, the set of functions one may call inside a signal handler is limited to very few of them. To highlight, mutexes (or other locking mechanisms) and memory allocation and deallocation are not allowed.

This library aims to solve some of the problems. It provides a global registry of actions performed on arrival of signals. It is possible to register multiple actions for the same signal and it is possible to remove the actions later on. If there was a previous signal handler when the first action for a signal is registered, it is chained (but the original one can't be removed).

Besides the basic registration of an arbitrary action, several helper actions are provided to cover the needs of the most common use cases.

## [signal-hook-registry (v1.0.1)](https://crates.io/crates/signal-hook-registry)

This is the backend crate for the signal-hook crate. The general direct use of this crate is discouraged. See the documentation for further details.

## [simple_logger (v0.5.0)](https://crates.io/crates/simple_logger)

A logger that prints all messages with a readable output format

```rust
#[macro_use]
extern crate log;
extern crate simple_logger;

fn main() {
    simple_logger::init().unwrap();

    warn!("This is an example message.");
}
```

output:

```bash
2015-02-24 01:05:20 WARN [logging_example] This is an example message.

```

## [slog-envlogger (v2.1.0)](https://crates.io/crates/slog-envlogger)

env_logger is a de facto standard Rust logger implementation, which allows controlling logging to stderr via the RUST_LOG environment variable.

This is a fork of env_logger that makes it work as a slog-rs drain.

Notable changes:

- Support for slog-stdlog to provide support for legacy info!(...) like statements.
- envlogger does not do any formatting anymore: slog-envlogger can be composed with any other slog-rs drains, so there's no point for it to provide it's own formatting. You can now output to a file, use JSON, have color output or any other future that slog ecosystem provides, controlling it via RUST_LOG environment var.

```rust
extern crate slog_stdlog;
extern crate slog_envlogger;

#[macro_use]
extern crate log;

fn main() {
    let _guard = slog_envlogger::init().unwrap();

    error!("error");
    info!("info");
    trace!("trace");
}
```

## [snow (v0.5.2)](https://crates.io/crates/snow)

An implementation of Trevor Perrin's Noise Protocol that is designed to be Hard To Fuck Up™.

```rust
let mut noise = snow::Builder::new("Noise_NN_25519_ChaChaPoly_BLAKE2s".parse()?)
                    .build_initiator()?;

let mut buf = [0u8; 65535];

// write first handshake message
noise.write_message(&[], &mut buf)?;

// receive response message
let incoming = receive_message_from_the_mysterious_ether();
noise.read_message(&incoming, &mut buf)?;

// complete handshake, and transition the state machine into transport mode
let mut noise = noise.into_transport_mode()?;
```

## [solicit (v0.4.4)](https://crates.io/crates/solicit)

An HTTP/2 implementation in Rust.

## [spin (v0.5.0)](https://crates.io/crates/spin)

Synchronization primitives based on spinning. They may contain data, are usable without `std`, and static initializers are available.

## [static_slice (v0.0.3)](https://crates.io/crates/static_slice)

Macro for creating static slices of arbitrary types.

## [string_cache (v0.7.3)](https://crates.io/crates/string_cache)

A string interning library for Rust, developed as part of the Servo project.

## [strsim (v0.9.2)](https://crates.io/crates/strsim)

Rust implementations of string similarity metrics:

- Hamming
- Levenshtein - distance & normalized
- Optimal string alignment
- Damerau-Levenshtein - distance & normalized
- Jaro and Jaro-Winkler - this implementation of Jaro-Winkler does not limit the common prefix length

The normalized versions return values between 0.0 and 1.0, where 1.0 means an exact match.

There are also generic versions of the functions for non-string inputs.

```rust
extern crate strsim;

use strsim::{hamming, levenshtein, normalized_levenshtein, osa_distance,
             damerau_levenshtein, normalized_damerau_levenshtein, jaro,
             jaro_winkler};

fn main() {
    match hamming("hamming", "hammers") {
        Ok(distance) => assert_eq!(3, distance),
        Err(why) => panic!("{:?}", why)
    }

    assert_eq!(levenshtein("kitten", "sitting"), 3);

    assert!((normalized_levenshtein("kitten", "sitting") - 0.571).abs() < 0.001);

    assert_eq!(osa_distance("ac", "cba"), 3);

    assert_eq!(damerau_levenshtein("ac", "cba"), 2);

    assert!((normalized_damerau_levenshtein("levenshtein", "löwenbräu") - 0.272).abs() <
            0.001);

    assert!((jaro("Friedrich Nietzsche", "Jean-Paul Sartre") - 0.392).abs() <
            0.001);

    assert!((jaro_winkler("cheeseburger", "cheese fries") - 0.911).abs() <
            0.001);
}
```

## [subtle (v2.1.0)](https://crates.io/crates/subtle)

Pure-Rust traits and utilities for constant-time cryptographic implementations. This library aims to be the Rust equivalent of Go’s crypto/subtle module.

## [term (v0.4.6)](https://crates.io/crates/term)

A Rust library for terminfo parsing and terminal colors.

```rust
extern crate term;
use std::io::prelude::*;

fn main() {
    let mut t = term::stdout().unwrap();

    t.fg(term::color::GREEN).unwrap();
    write!(t, "hello, ").unwrap();

    t.fg(term::color::RED).unwrap();
    writeln!(t, "world!").unwrap();

    t.reset().unwrap();
}
```

## [termcolor (v0.3.6)](https://crates.io/crates/termcolor)

A simple cross platform library for writing colored text to a terminal. This library writes colored text either using standard ANSI escape sequences or by interacting with the Windows console. Several convenient abstractions are provided for use in single-threaded or multi-threaded command line applications.

## [thread_local (v0.2.7)](https://crates.io/crates/thread_local)

This library provides the `ThreadLocal` and `CachedThreadLocal` types which allow a separate copy of an object to be used for each thread. This allows for per-object thread-local storage, unlike the standard library's `thread_local!` macro which only allows static thread-local storage.

## [thread-id (v3.3.0)](https://crates.io/crates/thread-id)

For diagnostics and debugging it can often be useful to get an ID that is different for every thread. Until Rust 1.14, the standard library did not expose a way to do that, hence this crate.

## [threshold_crypto (v0.3.1)](https://crates.io/crates/threshold_crypto)

A pairing-based threshold cryptosystem for collaborative decryption and signatures.

The threshold_crypto crate provides constructors for encrypted message handling. It utilizes the pairing elliptic curve library to create and enable reconstruction of public and private key shares.

In a network environment, messages are signed and encrypted, and key and signature shares are distributed to network participants. A message can be decrypted and authenticated only with cooperation from at least threshold + 1 nodes.

```rust
extern crate rand;
extern crate threshold_crypto;

use threshold_crypto::SecretKey;

/// Very basic secret key usage.
fn main() {
    let sk0 = SecretKey::random();
    let sk1 = SecretKey::random();

    let pk0 = sk0.public_key();

    let msg0 = b"Real news";
    let msg1 = b"Fake news";

    assert!(pk0.verify(&sk0.sign(msg0), msg0));
    assert!(!pk0.verify(&sk1.sign(msg0), msg0)); // Wrong key.
    assert!(!pk0.verify(&sk0.sign(msg1), msg0)); // Wrong message.
}
```

## [tiny-keccak (v1.4.3)](https://crates.io/crates/tiny-keccak)

An implementation of the FIPS-202-defined SHA-3 and SHAKE functions.

## [traitobject (v0.0.1)](https://crates.io/crates/traitobject)

Unsafe helpers for dealing with raw trait objects.

## [ttl_cache (v0.4.2)](https://crates.io/crates/ttl_cache)

A cache that will expire values after a TTL

## [typeable (v0.1.2)](https://crates.io/crates/typeable)

Exposes Typeable, for getting TypeIds at runtime.

## [typed-arena (v1.4.1)](https://crates.io/crates/typed-arena)

The arena, a fast but limited type of allocator

## [typenum (v1.10.0)](https://crates.io/crates/typenum)

Typenum is a Rust library for type-level numbers evaluated at compile time. It currently supports bits, unsigned integers, and signed integers.

Typenum depends only on libcore, and so is suitable for use on any platform!

```rust
use typenum::{Sum, Exp, Integer, N2, P3, P4};

type X = Sum<P3, P4>;
assert_eq!(<X as Integer>::to_i32(), 7);

type Y = Exp<N2, P3>;
assert_eq!(<Y as Integer>::to_i32(), -8);
```

## [unsigned-varint (v0.2.2)](https://crates.io/crates/unsigned-varint)

Unsigned varint encodes unsigned integers in 7-bit groups. The most significant bit (MSB) in each byte indicates if another byte follows (MSB = 1), or not (MSB = 0).

## [untrusted (v0.6.2)](https://crates.io/crates/untrusted)

Safe, fast, zero-panic, zero-crashing, zero-allocation parsing of untrusted inputs in Rust.

untrusted.rs is 100% Rust with no use of unsafe. It never uses the heap. No part of untrusted.rs's API will ever panic or cause a crash. It is #![no_std] and so it works perfectly with both libcore- and libstd- based projects. It does not depend on any crates other than libcore.

## [utf8-ranges (v0.1.3)](https://crates.io/crates/utf8-ranges)

This crate converts contiguous ranges of Unicode scalar values to UTF-8 byte ranges. This is useful when constructing byte based automata from Unicode. Stated differently, this lets one embed UTF-8 decoding as part of one's automaton.

```rust
extern crate utf8_ranges;

use utf8_ranges::Utf8Sequences;

fn main() {
    for range in Utf8Sequences::new('\u{0}', '\u{FFFF}') {
        println!("{:?}", range);
    }
}
```

output:

```bash
[0-7F]
[C2-DF][80-BF]
[E0][A0-BF][80-BF]
[E1-EC][80-BF][80-BF]
[ED][80-9F][80-BF]
[EE-EF][80-BF][80-BF]
```

## [uuid (v0.1.18)](https://crates.io/crates/uuid)

Generate and parse UUIDs.

Provides support for Universally Unique Identifiers (UUIDs). A UUID is a unique 128-bit number, stored as 16 octets. UUIDs are used to assign unique identifiers to entities without requiring a central allocating authority.

They are particularly useful in distributed systems, though can be used in disparate areas, such as databases and network protocols. Typically a UUID is displayed in a readable string form as a sequence of hexadecimal digits, separated into groups by hyphens.

The uniqueness property is not strictly guaranteed, however for all practical purposes, it can be assumed that an unintentional collision would be extremely unlikely.

```rust
use uuid::Uuid;

fn main() {
    let my_uuid =
        Uuid::parse_str("936DA01F9ABD4d9d80C702AF85C822A8").unwrap();
    println!("{}", my_uuid.to_urn());
}
```

## [wait-timeout (v0.2.0)](https://crates.io/crates/wait-timeout)

Rust crate for waiting on a Child process with a timeout specified.

## [x25519-dalek (v0.5.2)](https://crates.io/crates/x25519-dalek)

A pure-Rust implementation of x25519 elliptic curve Diffie-Hellman key exchange, with curve operations provided by curve25519-dalek.

This crate provides two levels of API: a bare byte-oriented x25519 function which matches the function specified in RFC7748, as well as a higher-level Rust API for static and ephemeral Diffie-Hellman.

First, Alice uses EphemeralSecret::new() and then PublicKey::from() to produce her secret and public keys:

```rust
extern crate rand_os;
extern crate x25519_dalek;

use rand_os::OsRng;

use x25519_dalek::EphemeralSecret;
use x25519_dalek::PublicKey;

let mut alice_csprng = OsRng::new().unwrap();
let     alice_secret = EphemeralSecret::new(&mut alice_csprng);
let     alice_public = PublicKey::from(&alice_secret);
```

Bob does the same:

```rust
let mut bob_csprng = OsRng::new().unwrap();
let     bob_secret = EphemeralSecret::new(&mut bob_csprng);
let     bob_public = PublicKey::from(&bob_secret);
```

Alice meows across the room, telling alice_public to Bob, and Bob loudly meows bob_public back to Alice. Alice now computes her shared secret with Bob by doing:

```rust
let alice_shared_secret = alice_secret.diffie_hellman(&bob_public);
```

Similarly, Bob computes a shared secret by doing:

```rust
let bob_shared_secret = bob_secret.diffie_hellman(&alice_public);
```

These secrets are the same:

```rust
assert_eq!(alice_shared_secret.as_bytes(), bob_shared_secret.as_bytes());
```

Voilá! Alice and Bob can now use their shared secret to encrypt their meows, for example, by using it to generate a key and nonce for an authenticated-encryption cipher.

## [yamux (v0.2.1)](https://crates.io/crates/yamux)

A stream multiplexer over reliable, ordered connections such as TCP/IP. Implements https://github.com/hashicorp/yamux/blob/master/spec.md.

## Other dependencies

- [rust-rocksdb](https://github.com/pingcap/rust-rocksdb.git)
- [bzip2-rs](https://github.com/alexcrichton/bzip2-rs.git)
- [lz4-rs](https://github.com/busyjay/lz4-rs.git)
- [rust-snappy](https://github.com/busyjay/rust-snappy.git)
- [zstd-rs](https://github.com/gyscos/zstd-rs.git)

All these are released with rocksdb.
