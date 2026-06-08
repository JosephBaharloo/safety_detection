"""
Eğitilmiş modeli projeye entegre etme scripti.

Kullanım:
    python scripts/integrate_trained_model.py --model-path path/to/trained/model.pth
    python scripts/integrate_trained_model.py --model-path runs/train/safety_detector/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def convert_pth_to_pt(pth_path: Path, pt_path: Path) -> bool:
    """
    .pth PyTorch formatını YOLOv8 .pt formatına dönüştür.
    
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        import torch
        from ultralytics import YOLO
        
        print(f"🔄 PyTorch formatından YOLOv8 formatına dönüştürülüyor...")
        
        # Önce YOLO olarak yükle (eğer mümkünse)
        try:
            model = YOLO(str(pth_path))
            model.save(str(pt_path))
            print(f"✅ YOLO ile dönüştürüldü")
            return True
        except Exception:
            # Eğer YOLO yükleyemezse, PyTorch olarak dene
            print(f"   ℹ️ YOLO yükleme başarısız, PyTorch yolu deneniyor...")
            state_dict = torch.load(pth_path, map_location="cpu")
            torch.save(state_dict, pt_path)
            print(f"✅ PyTorch ile kaydedildi (dikkat: YOLO uyumlu olmayabilir)")
            return True
            
    except ImportError as e:
        print(f"⚠️ Conversion için ultralytics veya torch gerekli: {e}")
        print(f"   Dosya olduğu gibi kopyalanacak...")
        return False
    except Exception as e:
        print(f"❌ Conversion hatası: {e}")
        return False


def validate_model(model_path: Path) -> bool:
    """
    Modelin YOLO tarafından yüklenebilip yüklenemeyeceğini kontrol et.
    """
    try:
        from ultralytics import YOLO
        
        print(f"🔍 Model validation yapılıyor...")
        model = YOLO(str(model_path))
        print(f"✅ Model YOLO tarafından yüklenebiliyor")
        
        # Model metadata'sını kontrol et
        if hasattr(model, 'names') and model.names:
            print(f"   Sınıflar: {model.names}")
        
        return True
    except Exception as e:
        print(f"⚠️ Model validation uyarısı: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eğitilmiş modeli projeye entegre et"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Eğitilmiş model dosyasının yolu (.pth veya .pt)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="best.pt",
        help="Modelin kaydedileceği isim (varsayılan: best.pt)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Model doğrulamasını atla"
    )

    args = parser.parse_args()
    model_path: Path = args.model_path
    output_name: str = args.name

    # Validasyonlar
    if not model_path.exists():
        print(f"❌ Hata: Model dosyası bulunamadı: {model_path}")
        return 1

    if not model_path.is_file():
        print(f"❌ Hata: Belirtilen yol bir dosya değil: {model_path}")
        return 1

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 0.1:
        print(f"❌ Hata: Model dosyası çok küçük ({file_size_mb:.2f}MB)")
        return 1

    # Hedef dizini belirle
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    output_path = models_dir / output_name

    # Format kontrolü ve dönüştürme
    if model_path.suffix.lower() == ".pth":
        print(f"📝 PyTorch formatı tespit edildi (.pth)")
        
        # .pth dosyasını .pt'ye dönüştür
        temp_pt_path = models_dir / "temp_converted.pt"
        if convert_pth_to_pt(model_path, temp_pt_path):
            model_path = temp_pt_path
        else:
            # Eğer conversion başarısız olursa, dosyayı olduğu gibi kopyala
            print(f"⚠️ Conversion başarısız, dosya olduğu gibi kopyalanacak")

    # Modeli kopyala
    try:
        print(f"📁 Model kopyalanıyor: {model_path.name} → {output_path}")
        shutil.copy2(model_path, output_path)
        
        # Temp dosyayı sil (varsa)
        if model_path.name == "temp_converted.pt":
            model_path.unlink()
        
        # Model doğrulaması
        if not args.no_validate:
            validate_model(output_path)
        
        print(f"✅ Model başarıyla entegre edildi!")
        print(f"   📍 Konum: {output_path}")
        print(f"   📊 Boyut: {file_size_mb:.2f}MB")
        print()
        print("📝 Sonraki adımlar:")
        print(f"   1. Uygulamayı yeniden başlat")
        print(f"   2. Logs'ta model yükleme durumunu kontrol et")
        print(f"   3. config/equipment_classes.yaml sınıflarını doğrula")
        return 0
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
