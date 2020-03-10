# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['WineClassify.py'],
             pathex=['C:\\Users\\freeverc\\PycharmProjects\\WineClassification'],
             binaries=[],
             datas=[],
			 hiddenimports=['cython',  'sklearn',  'sklearn.utils._cython_blas' ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [('v', None, 'OPTION')],
          exclude_binaries=True,
          name='WineClassify',
          debug=False,
          bootloader_ignore_signals=True,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='WineClassify')
