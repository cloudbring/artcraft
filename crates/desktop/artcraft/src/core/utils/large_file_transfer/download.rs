//! Implemented from https://github.com/huggingface/hf_transfer/blob/main/src/lib.rs

use crate::core::utils::large_file_transfer::error::Error;
use crate::core::utils::large_file_transfer::util::{exponential_backoff, BASE_WAIT_TIME, MAX_WAIT_TIME};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use log::info;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use reqwest::{StatusCode, Url};
use std::collections::HashMap;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::fs::OpenOptions;
use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tokio::time::sleep;

pub struct ProgressUpdate {
  pub complete: usize,
  pub total_length: usize,
}

/// max_files: Number of open file handles, which determines the maximum number of parallel downloads
/// parallel_failures:  Number of maximum failures of different chunks in parallel (cannot exceed max_files)
/// max_retries: Number of maximum attempts per chunk. (Retries are exponentially backed off + jitter)
///
/// The number of threads can be tuned by the environment variable `TOKIO_WORKER_THREADS` as documented in
/// https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.worker_threads
#[allow(clippy::too_many_arguments)]
pub async fn download_async<P: AsRef<Path>>(url: String, filename: NamedTempFile, final_filename: P, max_files: usize, chunk_size: usize, parallel_failures: usize, max_retries: usize, input_headers: Option<HashMap<String, String>>, maybe_progress_sender: Option<Sender<ProgressUpdate>>) -> Result<(), Error> {
  info!(">>> Downloading: {}", filename.as_ref().display());
  if max_files == 0 || chunk_size == 0 {
    return Err(Error::new_err("max_files and chunk_size must be nonzero".to_string()));
  }

  let client = reqwest::Client::builder()
    // https://github.com/hyperium/hyper/issues/2136#issuecomment-589488526
    .http2_keep_alive_timeout(Duration::from_secs(15))
    .build()?;

  let mut headers = HeaderMap::new();
  let mut auth_token = None;
  if let Some(input_headers) = input_headers {
    headers.reserve(input_headers.len());
    for (k, v) in input_headers {
      let name: HeaderName = k.try_into().map_err(|err| Error::new_err(format!("Invalid header: {err}")))?;
      let value: HeaderValue = AsRef::<str>::as_ref(&v).try_into().map_err(|err| Error::new_err(format!("Invalid header value: {err}")))?;
      if name == AUTHORIZATION {
        auth_token = Some(value);
      } else {
        headers.insert(name, value);
      }
    }
  };

  let response = if let Some(token) = auth_token.as_ref() { client.get(&url).header(AUTHORIZATION, token) } else { client.get(&url) }.headers(headers.clone()).header(RANGE, "bytes=0-0").send().await.map_err(|err| Error::new_err(format!("Error while downloading: {err}")))?;

  // Only call the final redirect URL to avoid overloading the Hub with requests and also
  // altering the download count
  let redirected_url = response.url().clone();
  if Url::parse(&url).map_err(|err| Error::new_err(format!("failed to parse url: {err}")))?.host() == redirected_url.host() {
    if let Some(token) = auth_token {
      headers.insert(AUTHORIZATION, token);
    }
  }

  if response.status() == StatusCode::OK {
    let length = response.headers().get(CONTENT_LENGTH).ok_or_else(|| Error::new_err("Range ignored and Content-Length missing".to_string()))?.to_str()?.parse::<usize>().map_err(|err| Error::new_err(format!("Invalid Content-Length: {err}")))?;
    download_sequential(response, filename.as_ref(), length).await?;
    send_progress(maybe_progress_sender.as_ref(), length, length);
    return persist_download(filename, final_filename.as_ref(), length);
  }

  let length = parse_content_range(response.headers().get(CONTENT_RANGE).ok_or_else(|| Error::new_err("Probe missing Content-Range".to_string()))?.to_str()?).and_then(|(start, stop, total)| (start == 0 && stop == 0).then_some(total)).ok_or_else(|| Error::new_err("Probe returned invalid Content-Range".to_string()))?;
  if response.status() != StatusCode::PARTIAL_CONTENT || response.bytes().await?.len() != 1 {
    return Err(Error::new_err("Probe returned invalid status or body length".to_string()));
  }

  info!("File {:?} is {:?} length", filename.as_ref(), length);

  let mut handles = FuturesUnordered::new();
  let semaphore = Arc::new(Semaphore::new(max_files));
  let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures.max(1)));
  let completed_bytes = AtomicUsize::new(0);

  for start in (0..length).step_by(chunk_size) {
    let url = redirected_url.to_string();
    let filename = filename.as_ref().to_owned();
    let client = client.clone();
    let headers = headers.clone();

    let stop = start.saturating_add(chunk_size).min(length) - 1;
    let semaphore = semaphore.clone();
    let parallel_failures_semaphore = parallel_failures_semaphore.clone();
    handles.push(tokio::spawn(async move {
      let permit = semaphore.acquire_owned().await.map_err(|err| Error::new_err(format!("Error while downloading: {err}")))?;
      let mut chunk = download_chunk(&client, &url, &filename, start, stop, length, headers.clone()).await;
      let mut i = 0;
      while let Err(dlerr) = chunk {
        if matches!(dlerr, Error::RangeIgnored) {
          return Err(dlerr);
        }
        if i >= max_retries {
          return Err(Error::new_err(format!("Failed after too many retries ({max_retries}): {dlerr}")));
        }
        let parallel_failure_permit = parallel_failures_semaphore.clone().acquire_owned().await.map_err(|err| Error::new_err(format!("Failed too many failures in parallel ({parallel_failures}): {dlerr} ({err})")))?;

        let wait_time = exponential_backoff(BASE_WAIT_TIME, i, MAX_WAIT_TIME);
        sleep(Duration::from_millis(wait_time as u64)).await;

        chunk = download_chunk(&client, &url, &filename, start, stop, length, headers.clone()).await;
        i += 1;
        drop(parallel_failure_permit);
      }
      drop(permit);
      chunk.map(|()| stop - start + 1).map_err(|err| Error::new_err(format!("Downloading error {err}")))
    }));
  }

  // Output the chained result
  while let Some(result) = handles.next().await {
    match result {
      Ok(Ok(size)) => {
        let complete = completed_bytes.fetch_add(size, Ordering::SeqCst) + size;
        send_progress(maybe_progress_sender.as_ref(), complete, length);
      },
      Ok(Err(Error::RangeIgnored)) => {
        abort_chunks(&mut handles).await;
        let response = client.get(redirected_url.clone()).headers(headers.clone()).send().await?.error_for_status()?;
        if response.status() != StatusCode::OK {
          return Err(Error::new_err("Sequential fallback did not return HTTP 200".to_string()));
        }
        download_sequential(response, filename.as_ref(), length).await?;
        send_progress(maybe_progress_sender.as_ref(), length, length);
        return persist_download(filename, final_filename.as_ref(), length);
      },
      Ok(Err(chunk_err)) => {
        abort_chunks(&mut handles).await;
        return Err(chunk_err);
      },
      Err(err) => {
        abort_chunks(&mut handles).await;
        return Err(Error::new_err(format!("Error while downloading: {err}")));
      },
    }
  }

  persist_download(filename, final_filename.as_ref(), length)
}

fn send_progress(maybe_sender: Option<&Sender<ProgressUpdate>>, complete: usize, total_length: usize) {
  if let Some(sender) = maybe_sender {
    let _ = sender.send(ProgressUpdate { complete, total_length });
  }
}

async fn download_chunk<P: AsRef<Path>>(client: &reqwest::Client, url: &str, filename: P, start: usize, stop: usize, total: usize, headers: HeaderMap) -> Result<(), Error> {
  let range = format!("bytes={start}-{stop}");
  let response = client.get(url).headers(headers).header(RANGE, range.clone()).send().await?;
  if response.status() == StatusCode::OK {
    return Err(Error::RangeIgnored);
  }
  response.error_for_status_ref()?;
  if response.status() != StatusCode::PARTIAL_CONTENT || response.headers().get(CONTENT_RANGE).and_then(|value| value.to_str().ok()).and_then(parse_content_range) != Some((start, stop, total)) {
    return Err(Error::new_err(format!("Invalid Content-Range for requested {range} (total {total})")));
  }
  let content = response.bytes().await?;
  if content.len() != stop - start + 1 {
    return Err(Error::new_err(format!("Wrong body length for requested {range}: got {} bytes", content.len())));
  }
  let mut file = OpenOptions::new().write(true).open(filename).await?;
  file.seek(SeekFrom::Start(start as u64)).await?;
  file.write_all(&content).await?;
  file.flush().await?;
  Ok(())
}

fn parse_content_range(value: &str) -> Option<(usize, usize, usize)> {
  let (unit, span) = value.split_once(' ')?;
  if unit != "bytes" {
    return None;
  }
  let (range, total) = span.split_once('/')?;
  let (start, stop) = range.split_once('-')?;
  let (start, stop, total) = (start.parse().ok()?, stop.parse().ok()?, total.parse().ok()?);
  (start <= stop && stop < total).then_some((start, stop, total))
}

async fn abort_chunks(handles: &mut FuturesUnordered<tokio::task::JoinHandle<Result<usize, Error>>>) {
  for handle in handles.iter() {
    handle.abort();
  }
  while handles.next().await.is_some() {}
}

async fn download_sequential(response: reqwest::Response, filename: &Path, length: usize) -> Result<(), Error> {
  let mut file = OpenOptions::new().write(true).truncate(true).open(filename).await?;
  let mut stream = response.bytes_stream();
  let mut received = 0usize;
  while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    received = received.checked_add(chunk.len()).ok_or_else(|| Error::new_err("Download exceeds expected length".to_string()))?;
    if received > length {
      return Err(Error::new_err(format!("Sequential download exceeds expected {length} bytes")));
    }
    file.write_all(&chunk).await?;
  }
  if received != length {
    return Err(Error::new_err(format!("Sequential download expected {length} bytes, received {received}")));
  }
  file.flush().await?;
  Ok(())
}

fn persist_download(filename: NamedTempFile, final_filename: &Path, length: usize) -> Result<(), Error> {
  let actual = filename.as_file().metadata()?.len();
  if actual != length as u64 {
    return Err(Error::new_err(format!("Final file size mismatch: expected {length}, got {actual}")));
  }
  let file = filename.persist(final_filename).map_err(|err| Error::new_err(format!("Could not rename to final filename: {err}")))?;
  file.sync_all()?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::{Read, Write};
  use std::net::{TcpListener, TcpStream};
  use std::sync::atomic::AtomicBool;
  use std::thread::{self, JoinHandle};

  const DATA: &[u8] = b"abcdefghijkl";

  #[derive(Clone, Copy)]
  enum Behavior {
    Valid,
    IgnoreProbe,
    IgnoreChunk,
    Short,
    WrongRange,
    WrongTotal,
    RetryTwice,
  }

  struct HttpStub {
    url: String,
    requests: Arc<std::sync::Mutex<Vec<String>>>,
    stopping: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
  }

  #[tokio::test]
  async fn range_ignoring_probe_downloads_sequentially() {
    let server = HttpStub::new(Behavior::IgnoreProbe);
    let (result, _directory, path) = download(&server, 2).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
    assert_eq!(server.requests.lock().unwrap().as_slice(), ["bytes=0-0"]);
  }

  #[tokio::test]
  async fn range_ignoring_chunk_falls_back_without_interleaving() {
    let server = HttpStub::new(Behavior::IgnoreChunk);
    let (result, _directory, path) = download(&server, 2).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
    assert!(server.requests.lock().unwrap().iter().any(|range| range.is_empty()));
  }

  #[tokio::test]
  async fn short_chunk_is_retried_then_rejected() {
    assert_invalid_chunk(Behavior::Short, "Wrong body length").await;
  }

  #[tokio::test]
  async fn wrong_content_range_is_retried_then_rejected() {
    assert_invalid_chunk(Behavior::WrongRange, "Invalid Content-Range").await;
  }

  #[tokio::test]
  async fn inconsistent_total_is_retried_then_rejected() {
    assert_invalid_chunk(Behavior::WrongTotal, "Invalid Content-Range").await;
  }

  #[tokio::test]
  async fn transient_chunk_failures_retry_without_corrupting_bytes() {
    let server = HttpStub::new(Behavior::RetryTwice);
    let (result, _directory, path) = download(&server, 2).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
    assert_eq!(server.requests.lock().unwrap().iter().filter(|range| *range == "bytes=4-7").count(), 3);
  }

  #[tokio::test]
  async fn valid_ranges_write_exact_file() {
    let server = HttpStub::new(Behavior::Valid);
    let (result, _directory, path) = download(&server, 0).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
  }

  #[tokio::test]
  async fn progress_counts_successful_chunks_only_even_after_retries() {
    let server = HttpStub::new(Behavior::RetryTwice);
    let (result, _directory, path, updates) = download_with_progress(&server, 2).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
    let completed: Vec<_> = updates.iter().map(|update| update.complete).collect();
    assert_eq!(completed, [4, 8, 12]);
    assert!(updates.iter().all(|update| update.total_length == DATA.len()));
    assert_eq!(server.requests.lock().unwrap().iter().filter(|range| *range == "bytes=4-7").count(), 3);
  }

  #[tokio::test]
  async fn final_partial_chunk_has_exact_range_and_progress() {
    let server = HttpStub::new(Behavior::Valid);
    let (result, _directory, path, updates) = download_with_progress_size(&server, 0, 5).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(std::fs::read(path).unwrap(), DATA);
    let mut previous = 0;
    let mut increments = updates
      .iter()
      .map(|update| {
        let increment = update.complete - previous;
        previous = update.complete;
        increment
      })
      .collect::<Vec<_>>();
    increments.sort_unstable();
    assert_eq!(increments, [2, 5, 5]);
    assert_eq!(previous, DATA.len());
    assert!(server.requests.lock().unwrap().iter().any(|range| range == "bytes=10-11"));
  }

  #[tokio::test]
  async fn sequential_fallback_reports_final_completed_bytes() {
    for behavior in [Behavior::IgnoreProbe, Behavior::IgnoreChunk] {
      let server = HttpStub::new(behavior);
      let (result, _directory, path, updates) = download_with_progress(&server, 2).await;
      assert!(result.is_ok(), "{result:?}");
      assert_eq!(std::fs::read(path).unwrap(), DATA);
      assert_eq!(updates.last().map(|update| (update.complete, update.total_length)), Some((DATA.len(), DATA.len())));
      assert!(updates.windows(2).all(|pair| pair[0].complete <= pair[1].complete));
    }
  }

  async fn assert_invalid_chunk(behavior: Behavior, expected: &str) {
    let server = HttpStub::new(behavior);
    let (result, _directory, path) = download(&server, 2).await;
    assert!(result.unwrap_err().to_string().contains(expected));
    assert!(!path.exists());
    assert!(server.requests.lock().unwrap().len() >= 4);
  }

  async fn download(server: &HttpStub, max_retries: usize) -> (Result<(), Error>, tempfile::TempDir, std::path::PathBuf) {
    let (result, directory, path, _) = download_with_progress(server, max_retries).await;
    (result, directory, path)
  }

  async fn download_with_progress(server: &HttpStub, max_retries: usize) -> (Result<(), Error>, tempfile::TempDir, std::path::PathBuf, Vec<ProgressUpdate>) {
    download_with_progress_size(server, max_retries, 4).await
  }

  async fn download_with_progress_size(server: &HttpStub, max_retries: usize, chunk_size: usize) -> (Result<(), Error>, tempfile::TempDir, std::path::PathBuf, Vec<ProgressUpdate>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("download.bin");
    let (sender, receiver) = std::sync::mpsc::channel();
    let result = download_async(server.url.clone(), NamedTempFile::new_in(directory.path()).unwrap(), &path, 3, chunk_size, 2, max_retries, None, Some(sender)).await;
    (result, directory, path, receiver.try_iter().collect())
  }

  impl HttpStub {
    fn new(behavior: Behavior) -> Self {
      let listener = TcpListener::bind("127.0.0.1:0").unwrap();
      listener.set_nonblocking(true).unwrap();
      let url = format!("http://{}", listener.local_addr().unwrap());
      let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
      let stopping = Arc::new(AtomicBool::new(false));
      let thread = {
        let requests = Arc::clone(&requests);
        let stopping = Arc::clone(&stopping);
        thread::spawn(move || {
          while !stopping.load(Ordering::Relaxed) {
            match listener.accept() {
              Ok((mut stream, _)) => serve(&mut stream, behavior, &requests),
              Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(5)),
              Err(err) => panic!("HTTP stub: {err}"),
            }
          }
        })
      };
      Self { url, requests, stopping, thread: Some(thread) }
    }
  }

  impl Drop for HttpStub {
    fn drop(&mut self) {
      self.stopping.store(true, Ordering::Relaxed);
      let _ = self.thread.take().unwrap().join();
    }
  }

  fn serve(stream: &mut TcpStream, behavior: Behavior, requests: &std::sync::Mutex<Vec<String>>) {
    stream.set_nonblocking(false).unwrap();
    let mut buffer = [0; 4096];
    let size = stream.read(&mut buffer).unwrap();
    let request = String::from_utf8_lossy(&buffer[..size]);
    let range = request.lines().find_map(|line| line.strip_prefix("range: ").or_else(|| line.strip_prefix("Range: "))).unwrap_or("").trim().to_string();
    let attempt = {
      let mut recorded = requests.lock().unwrap();
      recorded.push(range.clone());
      recorded.iter().filter(|item| **item == range).count()
    };
    let is_probe = range == "bytes=0-0";
    let ignored = matches!(behavior, Behavior::IgnoreProbe) && is_probe || matches!(behavior, Behavior::IgnoreChunk) && !is_probe;
    let (status, content_range, body) = if ignored || range.is_empty() {
      ("200 OK", None, DATA.to_vec())
    } else {
      let (start, stop) = range.strip_prefix("bytes=").unwrap().split_once('-').unwrap();
      let (start, stop) = (start.parse::<usize>().unwrap(), stop.parse::<usize>().unwrap());
      let mut body = DATA[start..=stop].to_vec();
      let mut response_start = start;
      let mut total = DATA.len();
      if !is_probe {
        match behavior {
          Behavior::Short => {
            body.pop();
          },
          Behavior::WrongRange => response_start += 1,
          Behavior::WrongTotal => total += 1,
          Behavior::RetryTwice if start == 4 && attempt <= 2 => {
            body.pop();
          },
          _ => {},
        }
      }
      ("206 Partial Content", Some(format!("bytes {response_start}-{stop}/{total}")), body)
    };
    let range_header = content_range.map(|header| format!("Content-Range: {header}\r\n")).unwrap_or_default();
    write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{range_header}Connection: close\r\n\r\n", body.len()).unwrap();
    stream.write_all(&body).unwrap();
  }
}
