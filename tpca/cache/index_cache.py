"""
Index Cache - caches parsed symbols to avoid re-parsing unchanged files.
"""
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config import TPCAConfig
from ..logging import StructuredLogger
from ..models import Symbol


class IndexCache:
    """
    Caches parsed symbols with per-file invalidation.
    
    Cache structure:
    .tpca_cache/
      index/
        <file_hash>.json  - cached symbols for each file
        manifest.json     - maps file paths to hashes and mtimes
    """
    
    def __init__(self, config: TPCAConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        
        self.cache_dir = Path(config.cache_dir) / 'index'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.cache_dir / 'manifest.json'
        self._manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """Load the cache manifest from disk."""
        if not self.manifest_path.exists():
            return {'files': {}, 'version': '1.0'}
        
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warn('manifest_load_failed', error=str(e))
            return {'files': {}, 'version': '1.0'}
    
    def _save_manifest(self):
        """Save the cache manifest to disk."""
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as e:
            self.logger.error('manifest_save_failed', error=str(e))
    
    def get(self, filepath: str) -> Optional[list[Symbol]]:
        """
        Get cached symbols for a file.
        
        Args:
            filepath: Path to source file
        
        Returns:
            List of Symbol objects, or None if not cached or invalid
        """
        if not self.config.cache_enabled:
            return None
        
        # Check if file is in manifest
        file_info = self._manifest['files'].get(filepath)
        if not file_info:
            return None
        
        # Check if file has been modified
        try:
            current_mtime = os.path.getmtime(filepath)
            if current_mtime > file_info['mtime']:
                self.logger.debug('cache_invalid', file=filepath, reason='modified')
                return None
        except OSError:
            return None
        
        # Load from cache
        cache_file = self.cache_dir / f"{file_info['hash']}.json"
        if not cache_file.exists():
            self.logger.debug('cache_miss', file=filepath, reason='file_not_found')
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Deserialize symbols
            symbols = [self._deserialize_symbol(s) for s in data['symbols']]
            self.logger.debug('cache_hit', file=filepath, symbol_count=len(symbols))
            return symbols
        
        except Exception as e:
            self.logger.warn('cache_load_failed', file=filepath, error=str(e))
            return None
    
    def set(self, filepath: str, symbols: list[Symbol]):
        """
        Cache symbols for a file.
        
        Args:
            filepath: Path to source file
            symbols: List of Symbol objects to cache
        """
        if not self.config.cache_enabled:
            return
        
        try:
            # Get file info
            mtime = os.path.getmtime(filepath)
            file_hash = self._hash_file(filepath)
            
            # Serialize symbols
            data = {
                'file': filepath,
                'cached_at': datetime.utcnow().isoformat(),
                'symbols': [self._serialize_symbol(s) for s in symbols]
            }
            
            # Write to cache
            cache_file = self.cache_dir / f"{file_hash}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update manifest
            self._manifest['files'][filepath] = {
                'hash': file_hash,
                'mtime': mtime,
                'symbol_count': len(symbols)
            }
            self._save_manifest()
            
            self.logger.debug('cache_set', file=filepath, symbol_count=len(symbols))
        
        except Exception as e:
            self.logger.warn('cache_write_failed', file=filepath, error=str(e))
    
    def invalidate(self, filepath: str):
        """
        Invalidate cache for a specific file.
        
        Args:
            filepath: Path to source file
        """
        if filepath in self._manifest['files']:
            file_info = self._manifest['files'].pop(filepath)
            
            # Remove cache file
            cache_file = self.cache_dir / f"{file_info['hash']}.json"
            if cache_file.exists():
                cache_file.unlink()
            
            self._save_manifest()
            self.logger.debug('cache_invalidated', file=filepath)
    
    def clear(self):
        """Clear the entire cache."""
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob('*.json'):
                if cache_file.name != 'manifest.json':
                    cache_file.unlink()
            
            # Reset manifest
            self._manifest = {'files': {}, 'version': '1.0'}
            self._save_manifest()
            
            self.logger.info('cache_cleared')
        
        except Exception as e:
            self.logger.error('cache_clear_failed', error=str(e))
    
    def _hash_file(self, filepath: str) -> str:
        """
        Generate a hash for a file based on its path and mtime.
        
        Args:
            filepath: Path to file
        
        Returns:
            Hash string
        """
        import hashlib
        mtime = os.path.getmtime(filepath)
        hash_input = f"{filepath}:{mtime}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _serialize_symbol(self, symbol: Symbol) -> dict:
        """Serialize a Symbol to JSON-compatible dict."""
        return {
            'id': symbol.id,
            'type': symbol.type,
            'name': symbol.name,
            'qualified_name': symbol.qualified_name,
            'file': symbol.file,
            'start_line': symbol.start_line,
            'end_line': symbol.end_line,
            'signature': symbol.signature,
            'docstring': symbol.docstring,
            'parent_class': symbol.parent_class,
            'bases': symbol.bases,
            'decorators': symbol.decorators,
            'pagerank': symbol.pagerank,
            'calls': symbol.calls,
        }
    
    def _deserialize_symbol(self, data: dict) -> Symbol:
        """Deserialize a Symbol from JSON dict."""
        return Symbol(
            id=data['id'],
            type=data['type'],
            name=data['name'],
            qualified_name=data['qualified_name'],
            file=data['file'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            signature=data['signature'],
            docstring=data.get('docstring', ''),
            parent_class=data.get('parent_class'),
            bases=data.get('bases', []),
            decorators=data.get('decorators', []),
            pagerank=data.get('pagerank', 0.0),
            calls=data.get('calls', []),
        )
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        return {
            'cached_files': len(self._manifest['files']),
            'cache_dir': str(self.cache_dir),
            'total_symbols': sum(f.get('symbol_count', 0)
                               for f in self._manifest['files'].values())
        }
