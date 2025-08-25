use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        fs::create_dir_all(path)?;
        tracing::info!("Created directory: {:?}", path);
    }
    Ok(())
}

pub fn backup_file<P: AsRef<Path>>(file_path: P) -> Result<PathBuf> {
    let file_path = file_path.as_ref();
    if !file_path.exists() {
        return Err(anyhow::anyhow!("File does not exist: {:?}", file_path));
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let backup_path = file_path.with_extension(format!(
        "{}.backup_{}",
        file_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or(""),
        timestamp
    ));

    fs::copy(file_path, &backup_path)?;
    tracing::info!("Created backup: {:?}", backup_path);
    Ok(backup_path)
}

pub fn safe_write<P: AsRef<Path>>(file_path: P, content: &str) -> Result<()> {
    let file_path = file_path.as_ref();
    
    // Create backup if file exists
    if file_path.exists() {
        backup_file(file_path)?;
    }

    // Ensure parent directory exists
    if let Some(parent) = file_path.parent() {
        ensure_directory(parent)?;
    }

    // Write to temporary file first
    let temp_path = file_path.with_extension("tmp");
    fs::write(&temp_path, content)?;

    // Atomic move to final location
    fs::rename(&temp_path, file_path)?;
    
    Ok(())
}

pub fn find_files_by_extension<P: AsRef<Path>>(
    dir: P, 
    extension: &str, 
    recursive: bool
) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let mut files = Vec::new();
    
    if !dir.exists() || !dir.is_dir() {
        return Ok(files);
    }

    find_files_recursive(dir, extension, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn find_files_recursive(
    dir: &Path,
    extension: &str,
    recursive: bool,
    files: &mut Vec<PathBuf>
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == extension.to_lowercase() {
                    files.push(path);
                }
            }
        } else if path.is_dir() && recursive {
            find_files_recursive(&path, extension, recursive, files)?;
        }
    }
    Ok(())
}

pub fn clean_old_files<P: AsRef<Path>>(
    dir: P,
    pattern: &str,
    max_age_days: u32
) -> Result<usize> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(0);
    }

    let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);
    let mut cleaned_count = 0;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.contains(pattern))
            .unwrap_or(false)
        {
            if let Ok(metadata) = entry.metadata() {
                if let Ok(modified) = metadata.modified() {
                    let modified_time = chrono::DateTime::<chrono::Utc>::from(modified);
                    if modified_time < cutoff {
                        fs::remove_file(&path)?;
                        cleaned_count += 1;
                        tracing::info!("Cleaned old file: {:?}", path);
                    }
                }
            }
        }
    }

    Ok(cleaned_count)
}

pub fn get_directory_size<P: AsRef<Path>>(path: P) -> Result<u64> {
    let path = path.as_ref();
    let mut total_size = 0;

    if path.is_file() {
        return Ok(path.metadata()?.len());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            total_size += path.metadata()?.len();
        } else if path.is_dir() {
            total_size += get_directory_size(&path)?;
        }
    }

    Ok(total_size)
}

pub fn compress_logs<P: AsRef<Path>>(log_dir: P) -> Result<()> {
    let log_dir = log_dir.as_ref();
    
    // This would implement log compression in a real application
    // For now, just clean old log files
    let cleaned = clean_old_files(log_dir, ".log", 7)?; // Keep logs for 7 days
    
    if cleaned > 0 {
        tracing::info!("Compressed/cleaned {} old log files", cleaned);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_ensure_directory() -> Result<()> {
        let temp_dir = tempdir()?;
        let test_dir = temp_dir.path().join("test_subdir");
        
        assert!(!test_dir.exists());
        ensure_directory(&test_dir)?;
        assert!(test_dir.exists());
        assert!(test_dir.is_dir());
        
        Ok(())
    }

    #[test]
    fn test_safe_write() -> Result<()> {
        let temp_dir = tempdir()?;
        let test_file = temp_dir.path().join("test.txt");
        
        safe_write(&test_file, "Hello, world!")?;
        
        assert!(test_file.exists());
        let content = fs::read_to_string(&test_file)?;
        assert_eq!(content, "Hello, world!");
        
        Ok(())
    }

    #[test]
    fn test_find_files_by_extension() -> Result<()> {
        let temp_dir = tempdir()?;
        
        // Create test files
        fs::write(temp_dir.path().join("test1.txt"), "content")?;
        fs::write(temp_dir.path().join("test2.txt"), "content")?;
        fs::write(temp_dir.path().join("test.log"), "content")?;
        
        let txt_files = find_files_by_extension(temp_dir.path(), "txt", false)?;
        assert_eq!(txt_files.len(), 2);
        
        let log_files = find_files_by_extension(temp_dir.path(), "log", false)?;
        assert_eq!(log_files.len(), 1);
        
        Ok(())
    }
}