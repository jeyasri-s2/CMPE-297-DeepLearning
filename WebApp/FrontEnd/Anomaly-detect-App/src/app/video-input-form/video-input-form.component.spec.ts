import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { VideoInputFormComponent } from './video-input-form.component';

describe('VideoInputFormComponent', () => {
  let component: VideoInputFormComponent;
  let fixture: ComponentFixture<VideoInputFormComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ VideoInputFormComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(VideoInputFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
